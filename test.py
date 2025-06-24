
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import copy
from tqdm import tqdm


class QuantumRNN:
    def __init__(self, n_hidden_qubits=3, n_input_qubits=1, learning_rate=0.02, n_layers=2):
        """
        初始化量子循环神经网络

        参数:
        n_hidden_qubits (int): 用于隐藏状态的量子比特数
        n_input_qubits (int): 用于输入编码的量子比特数
        learning_rate (float): 学习率
        n_layers (int): 变分层的数量
        """
        self.n_hidden_qubits = n_hidden_qubits
        self.n_input_qubits = n_input_qubits
        self.n_total_qubits = n_hidden_qubits + n_input_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))

        # 量子设备设置
        self.dev = qml.device("default.mixed", wires=self.n_total_qubits)

        # 初始化参数 - 为每个量子比特的每个层准备3个参数 (phi, theta, omega)
        self.weights = np.random.uniform(0, 2 * np.pi,
                                         size=(n_layers, self.n_total_qubits, 3),
                                         requires_grad=True)

        # 正则化参数
        self.reg_strength = 0.01

        # 构建模型
        self.build_model()

    def build_model(self):
        """构建量子模型"""

        # 更新隐藏状态的量子节点
        @qml.qnode(self.dev)
        def update_density_matrix(input_angle, hidden_rho, weights):
            """更新隐藏状态密度矩阵"""
            # 加载当前隐藏状态
            qml.QubitDensityMatrix(hidden_rho, wires=range(self.n_hidden_qubits))

            # 输入编码 (输入量子比特从隐藏量子比特之后开始)
            for i in range(self.n_input_qubits):
                qml.RY(input_angle, wires=self.n_hidden_qubits + i)

            # 应用参数化量子门 - 更强大的结构
            for layer in range(self.n_layers):
                # 旋转门
                for wire in range(self.n_total_qubits):
                    phi, theta, omega = weights[layer, wire]
                    qml.Rot(phi, theta, omega, wires=wire)

                # 灵活纠缠模式
                for i in range(0, self.n_total_qubits, 2):
                    if i + 1 < self.n_total_qubits:
                        qml.CNOT(wires=[i, i + 1])
                for i in range(1, self.n_total_qubits - 1, 2):
                    if i + 1 < self.n_total_qubits:
                        qml.CNOT(wires=[i, i + 1])

            # 返回更新后的隐藏状态
            return qml.density_matrix(wires=range(self.n_hidden_qubits))

        self.update_density_matrix = update_density_matrix

        # 整个QRNN的量子节点
        @qml.qnode(self.dev)
        def quantum_rnn(input_sequence, init_hidden_rho, weights):
            """处理完整输入序列并输出预测值"""
            current_rho = init_hidden_rho

            # 迭代处理输入序列
            for angle in input_sequence:
                new_rho = self.update_density_matrix(angle, current_rho, weights)
                current_rho = self.validate_and_fix_density_matrix(new_rho)

            # 加载最终隐藏状态
            qml.QubitDensityMatrix(current_rho, wires=range(self.n_hidden_qubits))

            # 定义可观测量的加权组合作为单个测量期望值
            H = qml.Hamiltonian(
                [0.5, 0.3, 0.2],
                [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)]
            )
            return qml.expval(H)

        self.quantum_rnn = quantum_rnn

    def get_init_hidden_state(self):
        """获取初始隐藏状态 (|0...0><0...0|)"""
        dim = 2 ** self.n_hidden_qubits
        rho = np.zeros((dim, dim), dtype=np.complex64)
        rho[0, 0] = 1.0
        return rho

    def validate_and_fix_density_matrix(self, rho):
        """确保密度矩阵合法（迹为1且半正定）"""
        # 1. 修复迹
        trace = np.trace(rho)
        if abs(trace - 1.0) > 1e-8:
            rho = rho / trace

        # 2. 确保半正定
        w, v = np.linalg.eigh(rho)
        if np.any(w < -1e-8):
            # 设置负特征值为0
            w = np.maximum(w, 0.0)
            # 重新归一化
            w = w / np.sum(w)
            # 重建矩阵
            rho = v @ np.diag(w) @ v.conj().T

        return rho

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=80, verbose=True):
        """
        训练模型

        参数:
        X_train (np.array): 训练输入序列
        y_train (np.array): 训练目标值
        X_val (np.array): 验证输入序列
        y_val (np.array): 验证目标值
        epochs (int): 训练轮数
        verbose (bool): 是否显示训练进度
        """
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)

        # 初始化隐藏状态
        init_rho = self.get_init_hidden_state()

        # 使用Adam优化器以获得更好的收敛性
        opt = qml.AdamOptimizer(stepsize=self.learning_rate, beta1=0.9, beta2=0.99)

        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(self.weights)
        start_time = time.time()

        for epoch in tqdm(range(epochs)):
            epoch_train_loss = 0
            shuffled_indices = np.random.permutation(X_train_scaled.shape[0])

            for i in shuffled_indices:
                # 定义带正则化的成本函数
                def cost(w):
                    pred = self.quantum_rnn(X_train_scaled[i], init_rho, w)

                    # 带正则化的损失
                    mse_loss = (pred - y_train[i]) ** 2
                    reg_loss = self.reg_strength * np.sum(w ** 2)
                    return mse_loss + reg_loss

                # 更新权重
                self.weights = opt.step(cost, self.weights)

                # 计算当前损失
                current_loss = cost(self.weights)
                epoch_train_loss += current_loss

            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / X_train_scaled.shape[0]
            train_losses.append(avg_train_loss)

            # 计算验证损失
            if X_val is not None:
                val_loss = 0
                for j in range(X_val_scaled.shape[0]):
                    val_loss += (self.quantum_rnn(X_val_scaled[j], init_rho, self.weights) - y_val[j]) ** 2
                avg_val_loss = val_loss / X_val_scaled.shape[0]
                val_losses.append(avg_val_loss)

                # 保存最佳权重
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_weights = copy.deepcopy(self.weights)
            else:
                avg_val_loss = None

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                elapsed = time.time() - start_time
                if avg_val_loss is not None:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {elapsed:.1f}s")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Time: {elapsed:.1f}s")
                start_time = time.time()

        # 恢复最佳权重
        if X_val is not None:
            self.weights = best_weights
            print(f"Best validation loss: {best_val_loss:.6f}")

        return train_losses, val_losses

    def predict(self, X):
        """
        使用训练好的模型进行预测

        参数:
        X (np.array): 输入序列 (形状: [n_samples, sequence_length])

        返回:
        np.array: 预测值
        """
        # 数据标准化 (使用训练时的scaler)
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)

        # 初始化隐藏状态
        init_rho = self.get_init_hidden_state()

        predictions = []
        for i in range(X_scaled.shape[0]):
            pred = self.quantum_rnn(X_scaled[i], init_rho, self.weights)
            predictions.append(pred)

        # 反向缩放预测值
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()

        return predictions

    def plot_results(self, train_losses, val_losses, y_test, predictions):
        """绘制训练结果和预测对比"""
        plt.figure(figsize=(16, 12))

        # 损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, 'b-', label='Training Loss')
        if val_losses:
            plt.plot(val_losses, 'r-', label='Validation Loss')
        plt.title('Quantum RNN Training & Validation Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # 预测对比
        plt.subplot(2, 1, 2)
        plt.plot(y_test, 'b-', label='True Values', alpha=0.8, linewidth=2)
        plt.plot(predictions, 'r--', label='Predictions', alpha=0.8, linewidth=2)
        plt.title('Time Series Prediction Results', fontsize=16)
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()



from get_data import get_senta_data, create_sequences
# 主程序
if __name__ == "__main__":
    np.random.seed(42)

    # 数据集参数
    seq_length = 6
    data=get_senta_data()
    n_samples=len(data)
    X, y = create_sequences(data, seq_length)

    # 划分训练集、验证集和测试集
    split1 = int(0.7 * n_samples)
    split2 = int(0.85 * n_samples)

    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]

    print(
        f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}, Test set size: {X_test.shape[0]}")

    # 初始化量子RNN模型
    print("\nInitializing Quantum RNN...")
    qrnn = QuantumRNN(n_hidden_qubits=3, n_input_qubits=1,
                      learning_rate=0.01, n_layers=3)

    # 训练模型
    print("Starting training...")
    start_train = time.time()
    train_losses, val_losses = qrnn.train(X_train, y_train, X_val, y_val, epochs=80)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.2f} seconds")

    # 在测试集上进行预测
    print("\nTesting model...")
    start_test = time.time()
    predictions = qrnn.predict(X_test)
    test_time = time.time() - start_test

    # 评估性能
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))

    print(f"Test Performance:")
    print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    print(f"Prediction time for test set: {test_time:.4f} seconds")

    # 绘制结果
    qrnn.plot_results(train_losses, val_losses, y_test, predictions)

    # 绘制完整的预测序列（测试部分）
    plt.figure(figsize=(14, 8))
    all_values = np.concatenate((y_train, y_val, y_test))
    test_start_idx = len(y_train) + len(y_val)
    test_range = range(test_start_idx, test_start_idx + len(y_test))

    plt.plot(range(len(all_values)), all_values, 'b-', label='True Values', alpha=0.7)
    plt.plot(test_range, predictions, 'ro-', label='Predictions', markersize=4, linewidth=1.5)

    # 添加区域标记
    plt.axvspan(0, len(y_train), color='green', alpha=0.05, label='Training Set')
    plt.axvspan(len(y_train), test_start_idx, color='orange', alpha=0.05, label='Validation Set')
    plt.axvspan(test_start_idx, len(all_values), color='red', alpha=0.05, label='Test Set')

    plt.title('Full Time Series with Predictions', fontsize=18)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

