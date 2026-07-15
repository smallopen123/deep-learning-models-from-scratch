"""训练 GAN 拟合二维高斯混合分布。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_models import Discriminator, Generator
from dl_models.synthetic import gaussian_mixture


def main() -> None:
    torch.manual_seed(42)
    noise_dim = 8
    generator, discriminator = Generator(noise_dim), Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(gaussian_mixture()), batch_size=64, shuffle=True)

    for epoch in range(20):
        for (real,) in loader:
            batch = len(real)
            fake = generator(torch.randn(batch, noise_dim))
            d_loss = loss_fn(discriminator(real), torch.ones(batch, 1)) + loss_fn(discriminator(fake.detach()), torch.zeros(batch, 1))
            d_optimizer.zero_grad(); d_loss.backward(); d_optimizer.step()

            generated = generator(torch.randn(batch, noise_dim))
            g_loss = loss_fn(discriminator(generated), torch.ones(batch, 1))
            g_optimizer.zero_grad(); g_loss.backward(); g_optimizer.step()
        print(f"epoch={epoch+1:02d} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")

    with torch.inference_mode():
        samples = generator(torch.randn(8, noise_dim))
    print("生成样本:\n", samples)


if __name__ == "__main__":
    main()
