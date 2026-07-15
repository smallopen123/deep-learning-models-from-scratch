"""无需下载的数据生成器，保证所有教程可离线快速上手。"""

import torch
from torch import Tensor


def tabular_classification(samples: int = 512) -> tuple[Tensor, Tensor]:
    x = torch.randn(samples, 2)
    y = ((x[:, 0] ** 2 + x[:, 1]) > 0.5).long()
    return x, y


def image_patterns(samples: int = 512, size: int = 28) -> tuple[Tensor, Tensor]:
    images = torch.zeros(samples, 1, size, size)
    labels = torch.arange(samples) % 4
    for index, label in enumerate(labels.tolist()):
        if label == 0: images[index, 0, :, size // 3] = 1       # 竖线
        elif label == 1: images[index, 0, size // 3, :] = 1     # 横线
        elif label == 2: images[index, 0].fill_diagonal_(1)      # 对角线
        else: images[index, 0, size // 3:2*size//3, size//3:2*size//3] = 1
    images += 0.15 * torch.randn_like(images)
    return images.clamp(0, 1), labels


def token_sequences(samples: int = 512, length: int = 12, vocab_size: int = 20) -> tuple[Tensor, Tensor]:
    tokens = torch.randint(1, vocab_size, (samples, length))
    labels = (tokens[:, : length // 2].sum(1) > tokens[:, length // 2:].sum(1)).long()
    return tokens, labels


def binary_vectors(samples: int = 512, features: int = 64) -> Tensor:
    prototypes = torch.randint(0, 2, (4, features)).float()
    ids = torch.randint(0, 4, (samples,))
    noisy = prototypes[ids] + 0.15 * torch.randn(samples, features)
    return noisy.clamp(0, 1)


def gaussian_mixture(samples: int = 512) -> Tensor:
    centers = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    ids = torch.randint(0, len(centers), (samples,))
    return centers[ids] + 0.35 * torch.randn(samples, 2)
