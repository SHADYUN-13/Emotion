import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMEmotionNet(nn.Module):
    def __init__(self, cnn_name='resnet18', lstm_hidden=256, lstm_layers=1, num_classes=3, dropout=0.5, pretrained=True):
        super().__init__()
        # 1. CNN主干，采用resnet18/34，去掉最后fc层
        if cnn_name == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
        elif cnn_name == 'resnet34':
            cnn = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError('Only support resnet18/34')
        feature_dim = cnn.fc.in_features
        modules = list(cnn.children())[:-1]  # 去除最后的全连接层
        self.cnn_backbone = nn.Sequential(*modules)

        # 2. LSTM部分
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 3. Dropout + 全连接
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feat = self.cnn_backbone(x)        # [B*T, feature_dim, 1, 1]
        feat = feat.view(B, T, -1)         # [B, T, feature_dim]
        lstm_out, _ = self.lstm(feat)      # [B, T, lstm_hidden]
        out = lstm_out[:, -1, :]           # 取最后一个时间步的输出
        out = self.dropout(out)
        logits = self.fc(out)              # [B, num_classes]
        return logits
