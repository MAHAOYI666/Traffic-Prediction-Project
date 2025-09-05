# evaluate.py

import torch
import os

from src import config
from src import utils
from src import data_processing
from src.models import cnn_transformer
from src.trainer import Trainer

def main():
    """
    加载已训练模型并进行评估的主流程
    """
    # 1. 设置
    utils.set_seed(config.SEED)
    model_name = "Ours_CNN_Transformer"
    model_params = config.OURS_MODEL_PARAMS
    print(f"开始评估已保存的模型: {model_name}")

    # 2. 加载数据 (使用与训练时完全相同的参数)
    _, test_loader, preprocessor, y_test_normalized = data_processing.get_dataloaders(
        config, model_params['lookback']
    )

    # 3. 初始化模型架构 (权重将在之后加载)
    model = cnn_transformer.CNNTransformerModel(
        num_layers=model_params['num_layers'],
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        dim_feedforward=model_params['dim_feedforward'],
        kernel_size=model_params['kernel_size']
    ).to(config.DEVICE)

    # 4. 实例化Trainer并加载模型权重
    # 只进行评估，optimizer和criterion在此处为None
    eval_trainer = Trainer(model, None, None, config.DEVICE, preprocessor, model_params)
    eval_trainer.load_model(model_name)

    # 5. 评估模型
    actuals_original = preprocessor.inverse_transform(y_test_normalized)
    metrics, (predictions_original, actuals_original) = eval_trainer.evaluate(
        test_loader, actuals_original_true_values=actuals_original
    )

    # 6. 可视化结果
    pred_plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_evaluation_plot.svg")
    utils.plot_predictions(
        actuals_original,
        predictions_original,
        model.model_type,
        pred_plot_path,
        scale_factor=config.PLOT_SCALE_FACTOR,
        font_size=config.FONT_SIZE
    )

    # 7. 效率分析
    utils.analyze_efficiency(
        model,
        config.DEVICE,
        model_params['lookback'],
        1, # feature_size
        config.BATCH_SIZE,
        model_params['beta'] # beta 作为最后一个位置参数
    )

if __name__ == "__main__":

    main()

