"""训练 Autoencoder 或 VAE 重建合成向量。"""

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_models import Autoencoder, VariationalAutoencoder
from dl_models.autoencoders import vae_loss
from dl_models.synthetic import binary_vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ae", "vae"], default="ae")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(42)
    data = binary_vectors()
    loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=True)
    model = Autoencoder() if args.model == "ae" else VariationalAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total = 0.0
        for (x,) in loader:
            output = model(x)
            loss = nn.functional.mse_loss(output[0], x) if args.model == "ae" else vae_loss(output[0], x, output[1], output[2])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(x)
        print(f"epoch={epoch+1:02d} loss={total/len(data):.4f}")


if __name__ == "__main__":
    main()
