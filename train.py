# train.py

import torch
import torch.nn as nn
import os

# 导入重构后的模块
from src import config
from src import utils
from src import data_processing
from src.models import cnn_transformer
from src.trainer import Trainer


def main():
    """
    主训练流程
    """
    # 1. 设置
    utils.set_seed(config.SEED)
    model_name = "Ours_CNN_Transformer"
    model_params = config.OURS_MODEL_PARAMS

    # 2. 加载数据
    # 使用与模型匹配的lookback
    train_loader, test_loader, preprocessor, y_test_normalized = data_processing.get_dataloaders(
        config, model_params['lookback']
    )

    # 3. 初始化模型、优化器和损失函数
    model = cnn_transformer.CNNTransformerModel(
        num_layers=model_params['num_layers'],
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        dim_feedforward=model_params['dim_feedforward'],
        kernel_size=model_params['kernel_size']
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=model_params['weight_decay']
    )
    criterion = nn.HuberLoss(delta=model_params['loss_delta'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 4. 实例化并运行Trainer
    trainer_instance = Trainer(model, optimizer, criterion, config.DEVICE, preprocessor, model_params)
    trainer_instance.train(train_loader, scheduler)

    # 5. 保存模型
    trainer_instance.save_model(model_name)

    # 6. 评估模型
    # 首先获取原始尺度的真实值，以便进行准确比较
    actuals_original = preprocessor.inverse_transform(y_test_normalized)
    metrics, (predictions_original, actuals_original) = trainer_instance.evaluate(
        test_loader, actuals_original_true_values=actuals_original
    )

    # 7. 可视化结果
    # 绘制损失曲线
    loss_plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_loss_curve.svg")
    utils.plot_loss_curve(trainer_instance.train_losses_history, f"{model.model_type} Loss Curve", loss_plot_path)

    # 绘制预测对比图
    pred_plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_prediction_plot.svg")
    utils.plot_predictions(
        actuals_original,
        predictions_original,
        model.model_type,
        pred_plot_path,
        scale_factor=config.PLOT_SCALE_FACTOR,
        font_size=config.FONT_SIZE
    )

    # 8. 效率分析
    # 对于需要额外参数的模型，在最后一个位置传入
    utils.analyze_efficiency(
        model,
        config.DEVICE,
        lookback=model_params['lookback'],
        batch_size=config.BATCH_SIZE,
        beta=model_params['beta']  # 传入beta参数
    )


if __name__ == "__main__":
    main()