from sklearn.base import BaseEstimator, ClassifierMixin
import pennylane as qml
from model_utils import *
from sklearn.preprocessing import MinMaxScaler
from get_data import get_senta_data,create_sequences

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

        dev = qml.device("default.mixed", wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit(params, input_val, hidden_state):
            # 设置初始量子态（密度矩阵）
            qml.QubitDensityMatrix(hidden_state, wires=range(self.n_qubits_))

            # 输入编码
            qml.RY(input_val, wires=0)

            # 参数化层
            for layer in range(self.n_layers_):
                # 单量子比特旋转
                for i in range(self.n_qubits_):
                    qml.RY(params[layer, i], wires=i)
                # 纠缠层
                for i in range(self.n_qubits_ - 1):
                    qml.CNOT(wires=[i, i + 1])

            # 返回整个系统的密度矩阵
            return qml.density_matrix(wires=range(self.n_qubits_))

        self.circuit = single_qubit_circuit

        def circuit(params, input_seq):
            # 初始化隐藏态：|0><0|
            hidden_state = jnp.zeros((2 ** self.n_qubits_, 2 ** self.n_qubits_), dtype=jnp.complex64)
            hidden_state = hidden_state.at[0, 0].set(1.0)  # |0><0| 态

            for x in input_seq:
                # 执行量子电路获取演化后的密度矩阵
                rho = single_qubit_circuit(params["weights"], x, hidden_state)

                # hidden_state = rho
                reduced_rho = ptrace(rho, self.n_qubits_)

                # 创建新量子比特的初始态 |0><0|
                new_qubit_state = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)

                # 克罗内克积扩展量子态
                hidden_state = jnp.kron(reduced_rho, new_qubit_state)

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

    # def output_transform(self, params, x):
    #     rho_image = jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1)
    #
    #     # 2. 卷积层 (需在外部初始化卷积核参数)
    #     conv_out = jax.lax.conv_general_dilated(
    #         rho_image[None, ...],  # 添加batch维度
    #         params['conv_weights'],
    #         window_strides=(2, 2),
    #         padding='SAME',
    #         dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    #     )
    #
    #     conv_out = jax.nn.relu(conv_out + params['conv_bias'])
    #     flattened = conv_out.reshape(-1)  # 展平
    #
    #     # 3. 全连接层
    #     output = jnp.dot(flattened,params['output_weights']) + params['output_bias']
    #     return output

    def output_transform(self, params, hidden_state):
        n_qubits = self.n_qubits_

        # 1. 计算所有单量子比特和双量子比特的期望值
        single_expectations = []
        for i in range(n_qubits):
            # 单量子比特Pauli Z期望值
            obs_matrix = qml.matrix(qml.PauliZ(i), wire_order=range(n_qubits))
            exp_val = jnp.trace(obs_matrix @ hidden_state)
            single_expectations.append(jnp.real(exp_val))

        pair_expectations = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # 双量子比特Pauli ZZ期望值
                obs_matrix = qml.matrix(qml.PauliZ(i) @ qml.PauliZ(j), wire_order=range(n_qubits))
                exp_val = jnp.trace(obs_matrix @ hidden_state)
                pair_expectations.append(jnp.real(exp_val))

        # 合并所有期望值
        features = jnp.concatenate([jnp.array(single_expectations),
                                    jnp.array(pair_expectations)])  # 形状: (n_qubits + C(n_qubits,2), )

        # 2. 通过MLP变换
        # 第二层: 线性变换 + 输出
        output = jnp.dot(features,params['output_weights2']) + params['output_bias2']

        return output

    def initialize_params(self):
        self.params_ = {"weights": 2 * jnp.pi * jax.random.uniform(shape=(self.n_layers_, self.n_qubits_), key=self.generate_key()),
                        'conv_weights': jax.random.normal(key=self.generate_key(), shape=(3, 3, 2, 8)) * 0.01,  # 3x3卷积核
                        'conv_bias': jnp.zeros(8),
                        'output_weights': jax.random.normal(key=self.generate_key(), shape=(2**(2*self.n_qubits_-2)*8, 1)) * 0.01,
                        'output_bias': jnp.zeros(1),

                        'output_weights2': jax.random.normal(key=self.generate_key(),
                                                            shape=(10, 1)) * 0.01,
                        'output_bias2': jnp.zeros(1)
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

    data=get_senta_data()
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
