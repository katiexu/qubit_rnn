from sklearn.base import BaseEstimator, ClassifierMixin
import pennylane as qml
from model_utils import *
from get_data import get_senta_data, create_sequences

jax.config.update("jax_enable_x64", True)

seq_length = 10
n_layers = 3
n_qubits = 4
max_steps=10000

def ptrace(rho,N):
    reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
    reduced_rho = jnp.einsum('ijik->jk', reshaped_rho,optimize=True)
    return reduced_rho
class SeparableVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            encoding_layers=1,
            learning_rate=0.001,
            batch_size=32,
            max_vmap=None,
            jit=True,
            max_steps=max_steps,
            random_state=42,
            scaling=1.0,
            convergence_interval=200,
            qnode_kwargs={"interface": "jax"},
            n_qubits_=4,
            layers=5
    ):
        # attributes that do not depend on data
        self.encoding_layers = encoding_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.convergence_interval = convergence_interval
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = n_qubits_
        self.n_layers_ = layers
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        # 使用 default.qubit 设备处理纯态
        dev = qml.device("default.qubit", wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit(params, x, hidden_state):
            # 使用 StatePrep 设置初始纯态
            qml.StatePrep(hidden_state, wires=range(self.n_qubits_))

            # 改进的输入编码：缩放并作用于所有量子比特
            for i in range(self.n_qubits_):
                qml.RY(x, wires=i)

            # 增强的变分层
            for layer in range(self.n_layers_):
                for i in range(self.n_qubits_):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                for i in range(self.n_qubits_):
                    for j in range(i + 1, self.n_qubits_):
                        qml.CNOT(wires=[i, j])
            # 返回演化后的态矢量
            return qml.state()

        self.circuit = single_qubit_circuit

        def circuit(params, input_seq):
            # 初始化隐藏态为 |0⟩ 的态矢量
            hidden_state = jnp.zeros(2 ** self.n_qubits_, dtype=jnp.complex64)
            hidden_state = hidden_state.at[0].set(1.0)  # |0⟩ 态

            for x in input_seq:
                # 执行量子电路获取演化后的态矢量
                state_vector = single_qubit_circuit(params["weights"], x, hidden_state)
                hidden_state = state_vector  # 更新隐藏态

            # 输出转换（根据需求调整）
            output = self.output_transform(params, hidden_state)
            return output

        if self.jit:
            circuit = jax.jit(circuit)
        self.forward = jax.vmap(circuit, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self, n_features):
        """Initialize attributes that depend on the number of features and the class labels.

        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """

        self.initialize_params()
        self.construct_model()

    def output_transform(self, params, hidden_state):
        # 提取量子态的实部和虚部作为特征
        features = jnp.concatenate([hidden_state.real, hidden_state.imag])
        # 增加非线性变换层
        hidden = jnp.dot(features, params['output_weights1']) + params['output_bias1']
        hidden = jax.nn.relu(hidden)  # 非线性激活
        output = jnp.dot(hidden, params['output_weights2']) + params['output_bias2']
        return output

    def initialize_params(self):
        self.params_ = {"weights": 2 * jnp.pi * jax.random.uniform(shape=(self.n_layers_, self.n_qubits_,2), key=self.generate_key()),
                        'output_weights1': jax.random.normal(key=self.generate_key(), shape=(2**self.n_qubits_*2, 2**self.n_qubits_)) * 0.01,
                        'output_bias1': jnp.zeros(2**self.n_qubits_),
                        'output_weights2': jax.random.normal(key=self.generate_key(),
                                                             shape=(2 ** self.n_qubits_, 1)) * 0.01,
                        'output_bias2': jnp.zeros(1),

                        }

    def fit(self, X, y):
        self.initialize(X.shape[1])

        optimizer = optax.adam

        def loss_fn(params, X, y):
            # we multiply by 6 because a relevant domain of the sigmoid function is [-6,6]
            vals = self.forward(params, X)
            return jnp.mean(jnp.mean((vals - y) ** 2))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        predictions = self.predict_proba(X)
        return jnp.squeeze(predictions)

    def predict_proba(self, X):
        predictions = self.chunked_forward(self.params_, X)
        return predictions


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    data = get_senta_data()
    time_steps = np.linspace(0, 10, len(data))


    X, y = create_sequences(data, seq_length)

    # 分割数据集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = SeparableVariationalClassifier(jit=True, max_vmap=32, layers=n_layers,n_qubits_=n_qubits)
    model.fit(X_train, y_train)
    train_predictions = np.array(model.predict(X_train))
    test_predictions = np.array(model.predict(X_test))

    # 可视化预测结果
    plt.figure(figsize=(14, 7))

    # 训练集预测
    plt.subplot(1, 2, 1)
    plt.plot(time_steps[seq_length:train_size + seq_length], y_train, label='Actual')
    plt.plot(time_steps[seq_length:train_size + seq_length], train_predictions, label='Predicted')
    plt.title('Training Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # 测试集预测
    plt.subplot(1, 2, 2)
    plt.plot(time_steps[train_size + seq_length:], y_test, label='Actual')
    plt.plot(time_steps[train_size + seq_length:], test_predictions, label='Predicted')
    plt.title('Test Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()
