"""RNN/LSTM/GRU 统一序列分类器。"""

from torch import Tensor, nn


class SequenceClassifier(nn.Module):
    """输入整数 token [B,L]，输出分类 logits [B,C]。"""

    def __init__(
        self, vocab_size: int = 20, embedding_dim: int = 16, hidden_dim: int = 32,
        num_classes: int = 2, cell: str = "lstm",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        cells = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        if cell not in cells:
            raise ValueError(f"cell 必须是 {tuple(cells)}")
        self.encoder = cells[cell](embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.cell = cell

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)       # [B,L] -> [B,L,E]
        _, hidden = self.encoder(embedded)
        if self.cell == "lstm":
            hidden = hidden[0]                  # LSTM 返回 (h_n, c_n)
        last_hidden = hidden[-1]                 # 最后一层: [B,H]
        return self.head(last_hidden)
