"""教学版全连接 GAN。生成二维点，便于 CPU 快速运行。"""

from torch import Tensor, nn


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 8, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim),
        )

    def forward(self, noise: Tensor) -> Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(0.2),
            nn.Linear(16, 1),  # 返回 logits；损失内部做 sigmoid
        )

    def forward(self, samples: Tensor) -> Tensor:
        return self.network(samples)
