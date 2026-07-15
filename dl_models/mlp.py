"""多层感知机（MLP）：全连接层与非线性激活的基础组合。"""

from torch import Tensor, nn


class MLP(nn.Module):
    """用于表格/向量分类的两层感知机。

    输入:  [batch, input_dim]
    输出:  [batch, num_classes]，是 logits，不是概率。
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # xW+b: [B,D] -> [B,H]
            nn.ReLU(),                         # 引入非线性
            nn.Linear(hidden_dim, num_classes),# [B,H] -> [B,C]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
