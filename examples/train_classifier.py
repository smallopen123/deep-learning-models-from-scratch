"""统一训练 MLP/CNN/ResNet/RNN/LSTM/GRU/Transformer。"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_models import MLP, SequenceClassifier, SimpleCNN, TinyResNet, TransformerClassifier
from dl_models.synthetic import image_patterns, tabular_classification, token_sequences


def build(name: str):
    if name == "mlp": return MLP(), tabular_classification()
    if name == "cnn": return SimpleCNN(), image_patterns()
    if name == "resnet": return TinyResNet(), image_patterns()
    if name in {"rnn", "lstm", "gru"}: return SequenceClassifier(cell=name), token_sequences()
    if name == "transformer": return TransformerClassifier(), token_sequences()
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn", "resnet", "rnn", "lstm", "gru", "transformer"], default="mlp")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (features, targets) = build(args.model)
    split = int(len(features) * 0.8)
    train_loader = DataLoader(TensorDataset(features[:split], targets[:split]), batch_size=32, shuffle=True)
    test_x, test_y = features[split:].to(device), targets[split:].to(device)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"epoch={epoch+1:02d} loss={total/split:.4f}")

    model.eval()
    with torch.inference_mode():
        accuracy = (model(test_x).argmax(1) == test_y).float().mean()
    print(f"model={args.model} device={device} test_accuracy={accuracy.item():.3f}")


if __name__ == "__main__":
    main()
