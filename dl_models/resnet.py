"""最小 ResNet：通过残差捷径学习 F(x)+x。"""

from torch import Tensor, nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 尺寸或通道变化时，用 1×1 卷积对齐捷径。
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.residual(inputs) + self.shortcut(inputs))


class TinyResNet(nn.Module):
    """适合 28×28/32×32 小图的教学版 ResNet。"""

    def __init__(self, num_classes: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.blocks = nn.Sequential(
            BasicBlock(16, 16),
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.blocks(self.stem(images))
        return self.head(self.pool(features).flatten(1))
