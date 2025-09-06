# src/data_processing.py

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """封装Box-Cox和Min-Max归一化/逆变换的类。"""

    def __init__(self, target_column):
        self.target_column = target_column
        self.boxcox_col = f"{target_column}_BoxCox"
        self.final_col = f"{target_column}_BoxCox_Normalized"
        self.lambda_ = None
        self.scaler = MinMaxScaler()

    def fit_transform(self, data_df):
        """在训练数据上拟合并进行变换。"""
        df = data_df.copy()
        # Box-Cox
        if (df[self.target_column] <= 0).any():
            print(f"警告: 列 '{self.target_column}' 包含非正值，已替换为0.001。")
            df.loc[df[self.target_column] <= 0, self.target_column] = 0.001
        transformed, self.lambda_ = stats.boxcox(df[self.target_column])
        df[self.boxcox_col] = transformed
        print(f"列 '{self.target_column}' 已应用Box-Cox变换，lambda={self.lambda_:.4f}。")
        # Min-Max
        df[self.final_col] = self.scaler.fit_transform(df[[self.boxcox_col]])
        print(f"列 '{self.boxcox_col}' 已进行Min-Max归一化。")
        return df

    def transform(self, data_df):
        """使用已拟合的参数变换新数据（验证/测试集）。"""
        df = data_df.copy()
        # Box-Cox
        if (df[self.target_column] <= 0).any():
            df.loc[df[self.target_column] <= 0, self.target_column] = 0.001
        df[self.boxcox_col] = stats.boxcox(df[self.target_column], lmbda=self.lambda_)
        # Min-Max
        df[self.final_col] = self.scaler.transform(df[[self.boxcox_col]])
        return df

    def inverse_transform(self, scaled_values):
        """将预测值逆变换回原始尺度。"""
        # Inverse Min-Max
        inversed_norm = self.scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
        # Inverse Box-Cox
        if self.lambda_ == 0:
            original_values = np.exp(inversed_norm)
        else:
            original_values = np.power(inversed_norm * self.lambda_ + 1, 1 / self.lambda_)
        return original_values


def load_data(file_path, target_col, timestamp_col, date_format):
    """从CSV加载和初步清洗数据。"""
    data = pd.read_csv(file_path)
    # 检查UL/DL是否存在，如果不存在则直接使用目标列
    if 'UL_bitrate' in data.columns and 'DL_bitrate' in data.columns:
        data = data[[timestamp_col, 'UL_bitrate', 'DL_bitrate']]
        data['Timestamp'] = pd.to_datetime(data[timestamp_col], format=date_format, errors='coerce')
        data[target_col] = data['UL_bitrate'] + data['DL_bitrate']
    else:
        data = data[[timestamp_col, target_col]]
        data['Timestamp'] = pd.to_datetime(data[timestamp_col], format=date_format, errors='coerce')

    data.sort_values(timestamp_col, inplace=True)
    data.set_index(timestamp_col, inplace=True)

    small_value_threshold = data[target_col].quantile(0.005)
    data.loc[data[target_col] < small_value_threshold, target_col] = 0.001
    return data[[target_col]]


def create_sequences(data_values, lookback):
    """从一维数组创建时间序列样本。"""
    X, y = [], []
    for i in range(len(data_values) - lookback):
        X.append(data_values[i:i + lookback])
        y.append(data_values[i + lookback])
    return np.array(X), np.array(y)


class TimeSeriesDataset(Dataset):
    """自定义PyTorch数据集。"""
    """PyTorch数据集标准形式。"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(config, lookback):
    """
    完整的数据处理流程，返回训练和测试的DataLoader及预处理器。
    """
    # 1. 加载和划分数据
    full_data = load_data(config.DATA_FILE, config.TARGET_COLUMN, config.TIMESTAMP_COL, config.DATE_FORMAT)
    train_size = int(len(full_data) * config.TRAIN_RATIO)
    train_df, test_df = full_data.iloc[:train_size].copy(), full_data.iloc[train_size:].copy()

    # 2. 数据预处理
    preprocessor = DataPreprocessor(config.TARGET_COLUMN)
    train_processed_df = preprocessor.fit_transform(train_df)
    test_processed_df = preprocessor.transform(test_df)

    train_values = train_processed_df[preprocessor.final_col].values
    test_values = test_processed_df[preprocessor.final_col].values

    # 3. 创建序列
    X_train, y_train = create_sequences(train_values, lookback)
    X_test, y_test = create_sequences(test_values, lookback)

    # 4. 创建数据集和加载器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"数据加载完成。训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")


    return train_loader, test_loader, preprocessor, y_test  # 返回y_test用于后续逆变换真实值
