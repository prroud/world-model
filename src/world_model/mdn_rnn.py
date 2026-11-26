"""LSTM mixture-density model for latent-state dynamics."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class MDNOutput(NamedTuple):
    """Mixture parameters for each timestep and the updated LSTM state."""

    mixture_logits: torch.Tensor
    means: torch.Tensor
    log_stds: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]


class MDNRNN(nn.Module):
    """Predict the next VAE latent distribution from latents and actions.

    Inputs use batch-first layout: latents ``(B, T, latent_dim)`` and actions
    ``(B, T, action_dim)``. Each mixture component predicts a diagonal Gaussian
    over the full next latent vector, with mixture weights shared by dimensions.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 3,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_mixtures: int = 5,
        min_log_std: float = -7.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if min(latent_dim, action_dim, hidden_size, num_layers, num_mixtures) < 1:
            raise ValueError("All dimensions and num_mixtures must be at least 1")
        if min_log_std >= max_log_std:
            raise ValueError("min_log_std must be lower than max_log_std")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mixture_head = nn.Linear(hidden_size, num_mixtures)
        self.mean_head = nn.Linear(hidden_size, num_mixtures * latent_dim)
        self.log_std_head = nn.Linear(hidden_size, num_mixtures * latent_dim)

    def initial_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zeroed LSTM hidden and cell states."""
        parameter = next(self.parameters())
        state_shape = (self.num_layers, batch_size, self.hidden_size)
        kwargs = {"device": device or parameter.device, "dtype": parameter.dtype}
        return torch.zeros(state_shape, **kwargs), torch.zeros(state_shape, **kwargs)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> MDNOutput:
        """Predict mixture parameters for each input timestep."""
        if latents.ndim != 3 or latents.shape[-1] != self.latent_dim:
            raise ValueError(f"latents must have shape (batch, time, {self.latent_dim})")
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must have shape (batch, time, {self.action_dim})")
        if latents.shape[:2] != actions.shape[:2]:
            raise ValueError("latents and actions must have matching batch and time dimensions")

        recurrent_input = torch.cat((latents, actions), dim=-1)
        features, next_hidden = self.lstm(recurrent_input, hidden)
        batch_size, sequence_length, _ = features.shape
        mixture_logits = self.mixture_head(features)
        means = self.mean_head(features).view(
            batch_size, sequence_length, self.num_mixtures, self.latent_dim
        )
        log_stds = self.log_std_head(features).view_as(means)
        log_stds = log_stds.clamp(self.min_log_std, self.max_log_std)
        return MDNOutput(mixture_logits, means, log_stds, next_hidden)

    @staticmethod
    def sample_next(output: MDNOutput, temperature: float = 1.0) -> torch.Tensor:
        """Sample a next latent vector from the final timestep's mixture.

        Temperature zero chooses the most likely component mean. For positive
        values, it adjusts both component selection and Gaussian variance.
        """
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        logits = output.mixture_logits[:, -1]
        means = output.means[:, -1]
        log_stds = output.log_stds[:, -1]
        if temperature == 0:
            component = logits.argmax(dim=-1)
            scale = 0.0
        else:
            component = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()
            scale = temperature**0.5
        gather_index = component[:, None, None].expand(-1, 1, means.shape[-1])
        selected_mean = means.gather(1, gather_index).squeeze(1)
        selected_log_std = log_stds.gather(1, gather_index).squeeze(1)
        if scale == 0.0:
            return selected_mean
        noise = torch.randn_like(selected_mean)
        return selected_mean + selected_log_std.exp() * scale * noise

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Advance one recurrent step and sample the next latent state."""
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent must have shape (batch, {self.latent_dim})")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must have shape (batch, {self.action_dim})")
        output = self(latent[:, None, :], action[:, None, :], hidden)
        return self.sample_next(output, temperature), output.hidden

    def step_with_uncertainty(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """Advance one step and also return mixture entropy and latent spread."""
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"latent must have shape (batch, {self.latent_dim})")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"action must have shape (batch, {self.action_dim})")
        output = self(latent[:, None, :], action[:, None, :], hidden)
        logits = output.mixture_logits[:, -1]
        weights = F.softmax(logits, dim=-1)
        entropy = -(weights * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if self.num_mixtures > 1:
            entropy = entropy / torch.log(logits.new_tensor(float(self.num_mixtures)))
        else:
            entropy = torch.zeros_like(entropy)

        means = output.means[:, -1]
        variances = torch.exp(2.0 * output.log_stds[:, -1])
        mixture_mean = (weights.unsqueeze(-1) * means).sum(dim=1)
        second_moment = (weights.unsqueeze(-1) * (variances + means.square())).sum(dim=1)
        predictive_std = (second_moment - mixture_mean.square()).clamp_min(0).sqrt()
        uncertainty = {
            "mixture_entropy": entropy.mean(),
            "mean_predictive_std": predictive_std.mean(),
        }
        return self.sample_next(output, temperature), output.hidden, uncertainty


def mixture_nll(
    output: MDNOutput, targets: torch.Tensor
) -> torch.Tensor:
    """Compute mean negative log likelihood for next-latent targets.

    ``targets`` has shape ``(batch, time, latent_dim)`` and is scored under
    the diagonal Gaussian mixture emitted at each corresponding timestep.
    """
    if targets.ndim != 3 or targets.shape != output.means.shape[:2] + (output.means.shape[-1],):
        raise ValueError("targets must match the output batch, time, and latent dimensions")
    log_stds = output.log_stds
    standardized = (targets.unsqueeze(2) - output.means) * torch.exp(-log_stds)
    log_prob_per_dim = -0.5 * standardized.square() - log_stds - 0.5 * torch.log(
        targets.new_tensor(2.0 * torch.pi)
    )
    component_log_prob = log_prob_per_dim.sum(dim=-1)
    log_mixture_weights = F.log_softmax(output.mixture_logits, dim=-1)
    log_prob = torch.logsumexp(log_mixture_weights + component_log_prob, dim=-1)
    return -log_prob.mean()
