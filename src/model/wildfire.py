import torch
import torch.nn as nn

N_FEATURES = 12

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=N_FEATURES, embed_dim=256):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 32),   # -> (B, 32, 32, 32)
            ConvBlock(32, 64),            # -> (B, 64, 16, 16)
            ConvBlock(64, 128),           # -> (B, 128, 8, 8)
            ConvBlock(128, 256),          # -> (B, 256, 4, 4)
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.proj    = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.proj(x)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=N_FEATURES, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=False
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.proj(out)


class WildfireFusionModel(nn.Module):
    def __init__(self, tab_dim=60, cnn_embed=256, lstm_embed=128, tab_embed=64):
        super().__init__()
        self.cnn  = CNNEncoder(embed_dim=cnn_embed)
        self.lstm = LSTMEncoder(hidden_dim=lstm_embed)

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, tab_embed),
            nn.ReLU(inplace=True),
        )

        fusion_in = cnn_embed + lstm_embed + tab_embed   # 448
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, img, seq, tab):
        cnn_emb  = self.cnn(img)
        lstm_emb = self.lstm(seq)
        tab_emb  = self.tab_mlp(tab)
        combined = torch.cat([cnn_emb, lstm_emb, tab_emb], dim=1)
        return self.fusion_head(combined).squeeze(1)


class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNNEncoder(embed_dim=256)
        self.lstm = LSTMEncoder(hidden_dim=128)
        self.head = nn.Sequential(
            nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, img, seq, tab):
        return self.head(torch.cat([self.cnn(img), self.lstm(seq)], dim=1)).squeeze(1)

