# run_comparison.py

import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# 导入重构后的模块
from src import config
from src import utils
from src import data_processing
from src.models import baselines
from src.trainer import Trainer


def main():
    """
    运行所有基线模型的对比实验。
    """
    utils.set_seed(config.SEED)

    # 1. 加载数据
    params = config.BASELINES_PARAMS
    train_loader, test_loader, preprocessor, y_test_normalized = data_processing.get_dataloaders(
        config, params['lookback']
    )
    actuals_original = preprocessor.inverse_transform(y_test_normalized)

    all_metrics = {}

    # 2. 定义要比较的深度学习模型
    dl_models_to_run = {
        "RNN": baselines.RNNModel(
            hidden_size=params['rnn_hidden_size'],
            num_layers=params['rnn_num_layers'],
            dropout=params['rnn_dropout']
        ),
        "LSTM": baselines.LSTMModel(
            hidden_size=params['rnn_hidden_size'],
            num_layers=params['rnn_num_layers'],
            dropout=params['rnn_dropout']
        ),
        "CNN": baselines.CNNModel(
            num_layers=params['cnn_num_layers'],
            out_channels=params['cnn_out_channels'],
            kernel_size=params['cnn_kernel_size'],
            lookback=params['lookback']
        ),
        "Transformer": baselines.TransformerModel(
            num_layers=params['transformer_num_layers'],
            d_model=params['transformer_d_model'],
            num_heads=params['transformer_num_heads'],
            dim_feedforward=params['transformer_dim_feedforward']
        )
    }

    # 3. 循环训练和评估深度学习模型
    for name, model in dl_models_to_run.items():
        print("\n" + "=" * 20 + f" 正在处理模型: {name} " + "=" * 20)
        model.to(config.DEVICE)

        # Transformer模型使用不同的weight_decay
        weight_decay = params.get('transformer_weight_decay', 0) if name == "Transformer" else 0
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)
        criterion = nn.HuberLoss(delta=params['loss_delta'])

        trainer_instance = Trainer(model, optimizer, criterion, config.DEVICE, preprocessor, {})
        trainer_instance.train(train_loader)

        metrics, (preds, actuals) = trainer_instance.evaluate(
            test_loader, actuals_original_true_values=actuals_original
        )
        all_metrics[name] = metrics

        # 为每个模型保存预测图
        plot_path = os.path.join(config.PLOTS_DIR, f"comparison_{name}_plot.svg")
        utils.plot_predictions(actuals, preds, name, plot_path)

    # 4. 训练和评估机器学习模型
    print("\n" + "=" * 20 + " 正在处理机器学习模型 " + "=" * 20)

    # 需要从数据加载器中重新获取序列数据
    X_train = train_loader.dataset.X.numpy()
    y_train = train_loader.dataset.y.numpy()
    X_test = test_loader.dataset.X.numpy()
    y_test = test_loader.dataset.y.numpy()

    ml_models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=config.SEED, n_jobs=-1),
        "KNN": KNeighborsRegressor(n_neighbors=int(np.sqrt(len(X_train))), n_jobs=-1)
    }

    for name, model in ml_models.items():
        print(f"训练 {name}...")
        model.fit(X_train, y_train)
        predictions_norm = model.predict(X_test)

        # 逆变换并计算指标
        predictions_orig = preprocessor.inverse_transform(predictions_norm)

        # 确保数据有效
        valid_indices = ~np.isnan(predictions_orig) & ~np.isnan(actuals_original)
        preds_valid = predictions_orig[valid_indices]
        actuals_valid = actuals_original[valid_indices]

        metrics = {
            'MAE': mean_absolute_error(actuals_valid, preds_valid),
            'RMSE': np.sqrt(mean_squared_error(actuals_valid, preds_valid)),
            'R2': r2_score(actuals_valid, preds_valid),
            'CC': utils.calculate_cc(actuals_valid, preds_valid),
            'SNR': utils.calculate_snr(actuals_valid, preds_valid)
        }
        all_metrics[name] = metrics
        print(f"{name} 评估结果: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

        # 绘图
        plot_path = os.path.join(config.PLOTS_DIR, f"comparison_{name}_plot.svg")
        utils.plot_predictions(actuals_valid, preds_valid, name, plot_path)

    # 5. 打印最终结果汇总
    results_df = pd.DataFrame(all_metrics).T  # .T转置，使模型名为行索引
    print("\n\n" + "=" * 50)
    print(" " * 15 + "模型性能对比汇总")
    print("=" * 50)
    print(results_df.to_string(float_format="%.4f"))
    print("=" * 50)


if __name__ == "__main__":
    main()