# tune_hyperparams.py

import optuna
import torch
import torch.nn as nn
import numpy as np

from src import config
from src import utils
from src.data_processing import load_data, DataPreprocessor, create_sequences, TimeSeriesDataset
from torch.utils.data import DataLoader
from src.models.cnn_transformer import CNNTransformerModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_optuna_dataloaders(cfg, lookback):
    """为Optuna搜索特别准备数据加载器，包含训练集和验证集。"""
    full_data = load_data(cfg.DATA_FILE, cfg.TARGET_COLUMN, cfg.TIMESTAMP_COL, cfg.DATE_FORMAT)

    # 使用 60/20/20 划分，与原始搜索脚本一致
    train_size = int(len(full_data) * 0.60)
    val_size = int(len(full_data) * 0.20)
    train_df = full_data.iloc[:train_size].copy()
    val_df = full_data.iloc[train_size:train_size + val_size].copy()

    preprocessor = DataPreprocessor(cfg.TARGET_COLUMN)
    train_processed = preprocessor.fit_transform(train_df)
    val_processed = preprocessor.transform(val_df)

    train_values = train_processed[preprocessor.final_col].values
    val_values = val_processed[preprocessor.final_col].values

    X_train, y_train = create_sequences(train_values, lookback)
    X_val, y_val = create_sequences(val_values, lookback)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, preprocessor


def objective(trial, train_loader, val_loader, preprocessor):
    """Optuna的目标函数。"""
    params = {
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256, 512]),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-4, 1e-3),
        'beta': trial.suggest_uniform('beta', 0.0, 1.0),
        'kernel_size': trial.suggest_int('kernel_size', 2, 7),
        'delta': trial.suggest_uniform('delta', 0.01, 1),
    }

    model = CNNTransformerModel(
        num_layers=params['num_layers'], d_model=params['d_model'],
        num_heads=params['num_heads'], dim_feedforward=params['dim_feedforward'],
        kernel_size=params['kernel_size']
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=params['weight_decay'])
    criterion = nn.HuberLoss(delta=params['delta'])

    best_val_rmse = float('inf')
    epochs_no_improve = 0

    for epoch in range(30):  # 每个试验训练最多30个epoch
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE).unsqueeze(-1), y_batch.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(X_batch, beta=params['beta'])
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        predictions_norm, actuals_norm = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.DEVICE).unsqueeze(-1)
                output = model(X_batch, beta=params['beta'])
                predictions_norm.append(output.cpu().numpy())
                actuals_norm.append(y_batch.cpu().numpy())

        predictions_norm = np.concatenate(predictions_norm).squeeze()
        actuals_norm = np.concatenate(actuals_norm).squeeze()

        preds_orig = preprocessor.inverse_transform(predictions_norm)
        actuals_orig = preprocessor.inverse_transform(actuals_norm)

        val_rmse = np.sqrt(mean_squared_error(actuals_orig, preds_orig))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:  # 早停
                break

    return best_val_rmse


def main():
    utils.set_seed(config.SEED)

    # 动态确定lookback的最大值，以确保数据足够
    # lookback将在objective函数中被Optuna选择
    dummy_lookback = 120  # 使用搜索范围的最大值
    train_loader, val_loader, preprocessor = get_optuna_dataloaders(config, dummy_lookback)

    study = optuna.create_study(direction='minimize')
    # Optuna的objective函数需要动态调整lookback，因此数据加载逻辑在函数内部
    # 此处传入的loader仅用于获取preprocessor
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, preprocessor), n_trials=100)

    print("\n最优超参数组合:")
    best_params = study.best_params
    print(best_params)
    print(f"最优验证 RMSE: {study.best_value:.6f}")
    print("\n请将这些最优参数更新到 config.py 文件中，然后运行 train.py 进行最终训练。")


if __name__ == "__main__":

    main()


