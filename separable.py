from sklearn.base import BaseEstimator, ClassifierMixin
import pennylane as qml
from model_utils import *
from sklearn.preprocessing import MinMaxScaler

jax.config.update("jax_enable_x64", True)

seq_length = 10
hidden_size = 8
n_layers = 3
n_qubits = 5

class SeparableVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            encoding_layers=1,
            learning_rate=0.001,
            batch_size=32,
            max_vmap=None,
            jit=True,
            max_steps=10000,
            random_state=42,
            scaling=1.0,
            convergence_interval=200,
            dev_type="default.qubit",
            qnode_kwargs={"interface": "jax"},
            hidden_size=hidden_size,
            n_qubits_=4,
            layers=5
    ):
        # attributes that do not depend on data
        self.encoding_layers = encoding_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.convergence_interval = convergence_interval
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.hidden_size = hidden_size

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

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit(params, input_val, hidden_state):
            # 将隐藏状态编码到量子电路
            for i in range(self.n_qubits_):
                qml.RY(hidden_state[i], wires=i)

            # 输入编码
            qml.RY(input_val, wires=0)

            # 参数化层
            for layer in range(self.n_layers_):
                # 单量子比特旋转
                for i in range(self.n_qubits_):
                    qml.RY(params[layer, i], wires=i)

                # 纠缠层
                for i in range(self.n_qubits_-1):
                    qml.CNOT(wires=[i, i + 1])

            # 返回期望值
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits_)]

        self.circuit = single_qubit_circuit

        def circuit(params, input_seq):
            hidden_state = jnp.zeros((self.n_qubits_, self.hidden_size))

            for x in input_seq:
                # 执行量子循环单元
                hidden_state = jnp.array(single_qubit_circuit(params["weights"], x, hidden_state))

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

    def output_transform(self, params, x):
        x = jax.nn.relu(jnp.dot(x, params["output_weights"]) + params["output_bias"])
        x = jnp.dot(params["output_weights2"], x) + params["output_bias2"]
        return jnp.squeeze(x)

    def initialize_params(self):
        # initialise the trainable parameters
        weights = (
                2
                * jnp.pi
                * jax.random.uniform(
            shape=(self.n_layers_, self.n_qubits_, self.hidden_size),
            key=self.generate_key(),
        )
        )

        output_weights = (
            jax.random.normal(shape=(self.hidden_size, 1), key=self.generate_key())
        )
        output_weights2 = (
            jax.random.normal(shape=(1, self.n_qubits_), key=self.generate_key())
        )
        output_bias = jax.random.normal(shape=(self.n_qubits_, 1), key=self.generate_key())
        output_bias2 = jax.random.normal(shape=(1,), key=self.generate_key())

        self.params_ = {"weights": weights,
                        "output_weights": output_weights,
                        "output_weights2": output_weights2,
                        "output_bias": output_bias,
                        "output_bias2": output_bias2}

    def fit(self, X, y):
        self.initialize(X.shape[1])

        optimizer = optax.adam

        def loss_fn(params, X, y):
            # we multiply by 6 because a relevant domain of the sigmoid function is [-6,6]
            vals = self.forward(params, X)
            y = jax.nn.relu(y)  # convert to 0,1
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

    # 生成时序数据
    np.random.seed(42)
    time_steps = np.linspace(0, 10, 500)
    data = np.sin(time_steps * 2 * np.pi * 0.2) + np.random.normal(0, 0.04, size=len(time_steps))

    # 数据归一化到 [0, π] 范围
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()


    # 准备数据
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)


    X, y = create_sequences(data_normalized, seq_length)

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
