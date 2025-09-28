"""
Neural Network Model Definitions.

Includes tabular deep learning model approximations—
LSTM for temporal dependencies,
1D CNN for feature extraction,
AutoEncoder for unsupervised representation learning,

All models output fraud prediction probabilities.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=29, hidden_dim=32, layer_dim=1, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_size, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, features) reshaped to (batch, seq_len=1, features) for LSTM
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class CNNModel(nn.Module):
    def __init__(self, input_channels=1, output_dim=1):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.layer2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 14, output_dim)  # Adjusted for expected input length
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch, channels=1, features)
        out = self.pool(torch.relu(self.layer1(x)))
        out = self.pool(torch.relu(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=29, encoding_dim=14):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def forward(self, x):
        return self.mlp(x)
