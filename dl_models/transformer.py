"""带位置编码的 Transformer Encoder 分类器。"""

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_length: int = 512) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        scales = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10_000.0) / model_dim))
        encoding = torch.zeros(max_length, model_dim)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.encoding[:, : inputs.size(1)]


class TransformerClassifier(nn.Module):
    """Encoder-only Transformer；输入 token [B,L]，输出 [B,C]。"""

    def __init__(
        self, vocab_size: int = 20, model_dim: int = 32, num_heads: int = 4,
        num_layers: int = 2, num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.position = PositionalEncoding(model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(model_dim, num_classes)
        self.scale = math.sqrt(model_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        padding_mask = tokens.eq(0)             # True 表示忽略 padding
        x = self.position(self.embedding(tokens) * self.scale)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled)
