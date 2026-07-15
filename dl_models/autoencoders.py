"""Autoencoder 与 Variational Autoencoder。"""

import torch
from torch import Tensor, nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim), nn.Sigmoid())

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encoder(inputs)
        return self.decoder(latent), latent


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 8) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.mean = nn.Linear(32, latent_dim)
        self.log_variance = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim), nn.Sigmoid())

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.shared(inputs)
        return self.mean(hidden), self.log_variance(hidden)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(std) * std

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(inputs)
        latent = self.reparameterize(mean, log_variance)
        return self.decoder(latent), mean, log_variance, latent


def vae_loss(reconstruction: Tensor, inputs: Tensor, mean: Tensor, log_variance: Tensor) -> Tensor:
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstruction, inputs, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return (reconstruction_loss + kl_divergence) / inputs.size(0)
