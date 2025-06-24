import numpy as np
from sklearn.preprocessing import MinMaxScaler
def generate_data():
    # 生成时序数据
    np.random.seed(42)
    time_steps = np.linspace(0, 10, 500)
    data = np.sin(time_steps * 2 * np.pi * 0.2) + np.random.normal(0, 0.04, size=len(time_steps))

    # 数据归一化到 [0, π] 范围
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return data_normalized
def get_senta_data():
    time_series_raw = np.load("./sk_Santa_Fe_2000.npy")
    K = len(time_series_raw)
    # normalize data
    min_ts = min(time_series_raw)
    max_ts = max(time_series_raw)
    time_series = (time_series_raw + np.abs(min_ts)) / (max_ts - min_ts)
    # flatten time series
    time_series = time_series.flatten()
    return time_series
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)
