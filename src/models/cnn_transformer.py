# src/models/cnn_transformer.py

import torch.nn as nn
import numpy as np
from .base_layers import PositionalEncoding


class CNNTransformerModel(nn.Module):
    """
    CNN-Transformer模型，根据审稿人建议修改的版本。
    """

    def __init__(self, feature_size=1, num_layers=3, d_model=128, num_heads=8, dim_feedforward=256, kernel_size=3):
        super(CNNTransformerModel, self).__init__()
        self.model_type = 'CNN-Transformer (Ours)'

        # CNN 部分
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=64, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.layer_norm1 = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Transformer 部分
        self.input_fc = nn.Linear(64, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()

        # 输出加权
        self.cnn_output_fc = nn.Linear(64, 1)

    def forward(self, src, beta=0.5):
        # CNN
        src_cnn = src.permute(0, 2, 1)
        src_cnn = self.conv1(src_cnn)
        src_cnn = src_cnn.permute(0, 2, 1)
        src_cnn = self.layer_norm1(src_cnn)
        src_cnn = src_cnn.permute(0, 2, 1)
        src_cnn = self.relu(src_cnn)
        src_cnn = self.pool(src_cnn)
        src_cnn = src_cnn.permute(0, 2, 1)

        # Transformer
        src_transformer = self.input_fc(src_cnn) * np.sqrt(self.transformer_encoder.layers[0].self_attn.embed_dim)
        src_transformer = self.pos_encoder(src_transformer)
        src_transformer = src_transformer.permute(1, 0, 2)
        output_transformer = self.transformer_encoder(src_transformer)
        output_transformer = output_transformer.permute(1, 0, 2)
        output_transformer = output_transformer[:, -1, :]
        output_transformer = self.decoder(output_transformer)

        # CNN直接输出
        output_cnn = self.cnn_output_fc(src_cnn[:, -1, :])

        # 加权组合
        combined_output = beta * output_cnn + (1 - beta) * output_transformer
        combined_output = self.softplus(combined_output)


        return combined_output
