"""Convolutional variational autoencoder for 64x64 RGB observations."""

from __future__ import annotations

import torch
from torch import nn


class VAE(nn.Module):
    """Encode and reconstruct normalized RGB images.

    Inputs and reconstructions use NCHW layout with values in ``[0, 1]``.
    The encoder produces a diagonal Gaussian distribution in latent space.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.to_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.from_latent = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent Gaussian parameters for a batch of NCHW images."""
        if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
            raise ValueError("images must have shape (batch, 3, 64, 64)")
        features = self.encoder(images)
        return self.to_mu(features), self.to_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from a diagonal Gaussian using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into normalized NCHW RGB images."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape (batch, {self.latent_dim})")
        return self.decoder(self.from_latent(latent))

    def forward(
        self, images: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, mean, and log variance.

        Set ``sample=False`` to decode the posterior mean, which is useful for
        deterministic reconstruction previews and latent dataset generation.
        """
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(latent), mu, logvar
