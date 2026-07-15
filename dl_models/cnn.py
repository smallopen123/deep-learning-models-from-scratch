"""小型卷积神经网络：卷积提取局部特征，池化压缩空间尺寸。"""

from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """接收 1×28×28 灰度图，输出类别 logits。"""

    def __init__(self, num_classes: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), # [B,C,28,28] -> [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                           # -> [B,16,14,14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                           # -> [B,32,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))
