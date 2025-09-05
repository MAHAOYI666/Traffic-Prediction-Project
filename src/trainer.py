# src/trainer.py

import torch
import torch.nn as nn
import copy
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from . import config
from . import utils
from .models.cnn_transformer import CNNTransformerModel  # 用于特殊处理beta参数


class Trainer:
    """
    负责训练、验证和评估模型的通用类。
    封装了训练循环、早停、学习率调度和模型保存逻辑。
    """

    def __init__(self, model, optimizer, criterion, device, preprocessor, model_params):
        """
        初始化Trainer。

        Args:
            model (torch.nn.Module): 要训练的模型。
            optimizer (torch.optim.Optimizer): 优化器。
            criterion (torch.nn.Module): 损失函数。
            device (torch.device): 训练设备 (CPU/GPU)。
            preprocessor (DataPreprocessor): 数据预处理器，用于逆变换评估指标。
            model_params (dict): 模型的超参数，可能包含beta等特定参数。
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.preprocessor = preprocessor
        self.model_params = model_params  # 存储模型的特定参数，例如CNN-Transformer的beta

        self.best_model_state = None
        self.best_loss = float('inf')
        self.train_losses_history = []
        self.epochs_no_improve = 0

        # 加载绘图风格
        utils.setup_plotting_style(config.FONT_SIZE, config.FONT_FAMILY)

    def _get_model_forward_args(self, is_eval=False):
        """
        根据模型类型和是否为评估模式，获取模型forward方法的额外参数。
        主要用于处理CNN-Transformer模型的beta参数。
        """
        # CNNTransformerModel需要beta参数
        if isinstance(self.model, CNNTransformerModel):
            return (self.model_params.get('beta', 0.5),)
        return ()  # 其他模型不需要额外参数

    def _train_epoch(self, train_loader):
        """
        执行一个训练epoch。
        """
        self.model.train()
        epoch_loss = 0
        forward_args = self._get_model_forward_args()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            X_batch = X_batch.unsqueeze(-1)  # 增加特征维度 (batch, seq_len, 1)

            self.optimizer.zero_grad()
            output = self.model(X_batch, *forward_args)
            loss = self.criterion(output.squeeze(), y_batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        return epoch_loss / len(train_loader.dataset)

    def train(self, train_loader, scheduler=None):
        """
        执行完整的训练过程，包含早停和学习率调度。

        Args:
            train_loader (DataLoader): 训练数据加载器。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。默认为None。
        """
        print(f"开始训练模型: {getattr(self.model, 'model_type', self.model.__class__.__name__)}...")
        print(f"训练设备: {self.device}")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss = self._train_epoch(train_loader)
            self.train_losses_history.append(train_loss)

            print(f"Epoch {epoch}/{config.NUM_EPOCHS}, 训练损失: {train_loss:.6f}")

            if scheduler:
                scheduler.step(train_loss)  # 通常scheduler基于验证损失，这里简化为基于训练损失

            # 早停逻辑 (基于训练损失进行简化，因为没有单独的验证集训练循环)
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                # print("模型性能提升，已保存最佳状态。")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= config.PATIENCE:
                    print(f"连续 {config.PATIENCE} 个 epoch 性能未提升，触发早停。")
                    break

        # 加载最佳模型状态
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print("已加载性能最佳的模型状态。")
        else:
            print("未找到更好的模型状态，使用最后一次训练的模型。")

    def evaluate(self, test_loader, actuals_original_true_values=None):
        """
        在测试集上评估模型，并返回预测值、真实值和各项评估指标。

        Args:
            test_loader (DataLoader): 测试数据加载器。
            actuals_original_true_values (np.ndarray, optional): 原始尺度的真实值，用于评估。
                                                                如果提供，则使用此值，否则从preprocessor逆变换。
        Returns:
            dict: 包含MAE, RMSE, R2, CC, SNR等指标。
            tuple: (predictions_original, actuals_original) 原始尺度的预测值和真实值。
        """
        self.model.eval()
        predictions_normalized = []
        actuals_normalized = []
        forward_args = self._get_model_forward_args(is_eval=True)

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device).unsqueeze(-1)
                output = self.model(X_batch, *forward_args)
                predictions_normalized.append(output.cpu().numpy())
                actuals_normalized.append(y_batch.cpu().numpy())

        predictions_normalized = np.concatenate(predictions_normalized).squeeze()
        actuals_normalized = np.concatenate(actuals_normalized).squeeze()

        # 逆变换预测值和真实值
        predictions_original = self.preprocessor.inverse_transform(predictions_normalized)

        if actuals_original_true_values is not None:
            actuals_original = actuals_original_true_values
        else:
            # 如果没有提供原始真实值，则使用preprocessor进行逆变换
            actuals_original = self.preprocessor.inverse_transform(actuals_normalized)

        # 确保预测值和真实值长度一致且无NaN
        valid_indices = ~np.isnan(predictions_original) & ~np.isnan(actuals_original)
        predictions_original = predictions_original[valid_indices]
        actuals_original = actuals_original[valid_indices]

        # 计算评估指标
        mse = mean_squared_error(actuals_original, predictions_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_original, predictions_original)
        r2 = r2_score(actuals_original, predictions_original)
        snr = utils.calculate_snr(actuals_original, predictions_original)
        cc = utils.calculate_cc(actuals_original, predictions_original)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CC': cc,
            'SNR': snr
        }

        model_name = getattr(self.model, 'model_type', self.model.__class__.__name__)
        print(f"\n--- {model_name} 模型评估结果 ---")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, CC: {cc:.4f}, SNR: {snr:.4f} dB")

        return metrics, (predictions_original, actuals_original)

    def save_model(self, model_name, path=config.SAVED_MODELS_DIR):
        """保存训练好的模型。"""
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{model_name}_best_model.pth")
        torch.save(self.model.state_dict(), file_path)
        print(f"模型已保存至: {file_path}")

    def load_model(self, model_name, path=config.SAVED_MODELS_DIR):
        """加载已保存的模型。"""
        file_path = os.path.join(path, f"{model_name}_best_model.pth")
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"模型已从 {file_path} 加载。")
        else:
            print(f"警告: 未找到模型文件 {file_path}。")