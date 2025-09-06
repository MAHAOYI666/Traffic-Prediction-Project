# src/config.py

import torch

# --- 1. 路径和基本设置 ---
DATA_FILE = 'data/HSR.csv'  # 数据文件路径
PLOTS_DIR = 'results/plots/'  # 保存绘图的目录
SAVED_MODELS_DIR = 'results/saved_models/'  # 保存模型的目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# --- 2. 数据处理参数 ---
TARGET_COLUMN = 'Throughput'  # 目标列
TRAIN_RATIO = 0.8  # 训练集比例
TIMESTAMP_COL = 'Timestamp'
DATE_FORMAT = '%Y.%m.%d_%H.%M.%S.%f'

# --- 3. 训练参数 ---
BATCH_SIZE = 64
NUM_EPOCHS = 50
PATIENCE = 10  # 早停耐心轮数
LEARNING_RATE = 0.001

# --- 4. 绘图设置 ---
FONT_SIZE = 35
FONT_FAMILY = 'Times New Roman'
PLOT_SCALE_FACTOR = 1e7  # 绘图时的y轴缩放因子

# --- 5. 模型超参数 ---

OURS_MODEL_PARAMS = {
    'lookback': 95,
    'num_layers': 3,
    'd_model': 128,
    'num_heads': 8,
    'dim_feedforward': 256,
    'weight_decay': 0.0001333,  # L2 正则化
    'beta': 0.3094,            # CNN和Transformer输出的加权因子
    'kernel_size': 7,
    'loss_delta': 0.6563       # HuberLoss的delta
}

# BiLSTM 模型
BILSTM_PARAMS = {
    'lookback': 40,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'weight_decay': 0.0001,
    'loss_delta': 0.5
}

# 基线模型通用参数 
BASELINES_PARAMS = {
    'lookback': 90,
    'loss_delta': 1.0,
    # RNN/LSTM
    'rnn_hidden_size': 64,
    'rnn_num_layers': 2,
    'rnn_dropout': 0.2,
    # CNN
    'cnn_num_layers': 2,
    'cnn_out_channels': 64,
    'cnn_kernel_size': 3,
    'cnn_dropout': 0.2,
    # Transformer
    'transformer_num_layers': 2,
    'transformer_d_model': 128,
    'transformer_num_heads': 8,
    'transformer_dim_feedforward': 256,
    'transformer_weight_decay': 0.0001333

}
