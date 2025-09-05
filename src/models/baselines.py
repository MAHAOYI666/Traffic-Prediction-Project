# src/models/baselines.py

import torch
import torch.nn as nn
from .base_layers import PositionalEncoding

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(RNNModel, self).__init__()
        self.model_type = 'RNN'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return self.softplus(out)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.model_type = 'LSTM'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.softplus(out)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.model_type = 'Bi-LSTM'
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])
        return self.softplus(prediction)

class CNNModel(nn.Module):
    def __init__(self, feature_size=1, num_layers=2, out_channels=64, kernel_size=3, lookback=50):
        super(CNNModel, self).__init__()
        self.model_type = 'CNN'
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.relu = nn.ReLU()
        in_channels, current_lookback = feature_size, lookback
        for _ in range(num_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(nn.LayerNorm(out_channels))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
            current_lookback = current_lookback // 2
        self.fc = nn.Linear(out_channels * current_lookback, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            x = self.norm_layers[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.relu(x)
            x = self.pool_layers[i](x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return self.softplus(x)

class TransformerModel(nn.Module):
    def __init__(self, feature_size=1, num_layers=2, d_model=128, num_heads=8, dim_feedforward=256):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()

    def forward(self, src):
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return self.softplus(output)