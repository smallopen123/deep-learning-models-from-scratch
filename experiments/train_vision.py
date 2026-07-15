"""用 FashionMNIST/CIFAR-10 验证 MLP、CNN、ResNet。首次运行会下载数据。"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dl_models import MLP, SimpleCNN, TinyResNet


def prepare(dataset_name: str, limit: int | None):
    root = Path("data")
    if dataset_name == "fashion_mnist":
        transform = transforms.ToTensor()
        train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
        channels, size = 1, 28
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
        train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root, train=False, download=True, transform=transform)
        channels, size = 3, 32
    if limit:
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train), generator=generator)[: min(limit, len(train))]
        train = Subset(train, indices)
        test = Subset(test, range(min(2000, len(test))))
    return train, test, channels, size


def build(model_name: str, channels: int, size: int) -> nn.Module:
    if model_name == "mlp":
        return nn.Sequential(nn.Flatten(), MLP(channels * size * size, 128, 10))
    if model_name == "cnn":
        if size != 28: raise ValueError("教学版 CNN 分类头固定为 28×28，请用 FashionMNIST")
        return SimpleCNN(num_classes=10, in_channels=channels)
    return TinyResNet(num_classes=10, in_channels=channels)


def evaluate(model, loader, device):
    model.eval(); correct = total = 0; loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device); logits = model(x)
            loss_sum += loss_fn(logits, y).item() * len(x)
            correct += (logits.argmax(1) == y).sum().item(); total += len(x)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn", "resnet"], required=True)
    parser.add_argument("--dataset", choices=["fashion_mnist", "cifar10"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=12000, help="0 表示使用全部训练集")
    args = parser.parse_args(); torch.manual_seed(42)
    train, test, channels, size = prepare(args.dataset, args.limit or None)
    train_loader = DataLoader(train, 64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, 256, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(args.model, channels, size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * len(x)
        test_loss, accuracy = evaluate(model, test_loader, device)
        print(f"epoch={epoch+1:02d} train_loss={running/len(train):.4f} test_loss={test_loss:.4f} accuracy={accuracy:.3f}")
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{args.model}_{args.dataset}.pt")


if __name__ == "__main__": main()
