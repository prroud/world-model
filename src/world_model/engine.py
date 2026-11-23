"""Inference wrapper that joins the VAE decoder and MDN-RNN dynamics model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from world_model.mdn_rnn import MDNRNN
from world_model.vae import VAE


class WorldEngine:
    """Generate CarRacing-like frames from actions using trained neural models.

    The engine needs encoded real episodes as reset seeds. After reset, each
    call to ``step`` advances only the MDN-RNN and VAE; Gymnasium is not used
    during dream inference.
    """

    def __init__(
        self,
        vae_checkpoint: str | Path = "artifacts/vae/vae.pt",
        dynamics_checkpoint: str | Path = "artifacts/mdn_rnn/mdn_rnn.pt",
        seed_dir: str | Path = "data/latents",
        device: str = "auto",
        temperature: float = 1.0,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        device_name = device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.seed_dir = Path(seed_dir)
        self._seed_files = sorted(self.seed_dir.glob("episode_*.npz"))
        if not self._seed_files:
            raise FileNotFoundError(f"No encoded seed episodes found in {self.seed_dir}")

        vae_payload = self._load_checkpoint(Path(vae_checkpoint))
        self.vae = VAE(latent_dim=int(vae_payload["latent_dim"]))
        self.vae.load_state_dict(vae_payload["state_dict"])
        self.vae.to(self.device).eval()

        dynamics_payload = self._load_checkpoint(Path(dynamics_checkpoint))
        self.dynamics = MDNRNN(
            latent_dim=int(dynamics_payload["latent_dim"]),
            action_dim=int(dynamics_payload["action_dim"]),
            hidden_size=int(dynamics_payload["hidden_size"]),
            num_layers=int(dynamics_payload["num_layers"]),
            num_mixtures=int(dynamics_payload["num_mixtures"]),
        )
        if self.dynamics.latent_dim != self.vae.latent_dim:
            raise ValueError("VAE and MDN-RNN checkpoints use different latent dimensions")
        self.dynamics.load_state_dict(dynamics_payload["state_dict"])
        self.dynamics.to(self.device).eval()

        self._temperature = float(temperature)
        self._rng = np.random.default_rng()
        self._latent: torch.Tensor | None = None
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def readiness(
        vae_checkpoint: str | Path = "artifacts/vae/vae.pt",
        dynamics_checkpoint: str | Path = "artifacts/mdn_rnn/mdn_rnn.pt",
        seed_dir: str | Path = "data/latents",
    ) -> dict[str, object]:
        """Check whether the default inference assets are present without loading them."""
        vae_path = Path(vae_checkpoint)
        dynamics_path = Path(dynamics_checkpoint)
        seed_path = Path(seed_dir)
        seed_files = sorted(seed_path.glob("episode_*.npz")) if seed_path.is_dir() else []
        checks = {
            "vae_checkpoint": vae_path.is_file(),
            "dynamics_checkpoint": dynamics_path.is_file(),
            "seed_episodes": bool(seed_files),
        }
        return {
            "status": "ready" if all(checks.values()) else "not_ready",
            "checks": checks,
            "seed_episode_count": len(seed_files),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    @property
    def latent(self) -> np.ndarray:
        """Return a copy of the current latent vector for inspection."""
        if self._latent is None:
            raise RuntimeError("Call reset() before reading the current latent state")
        return self._latent[0].detach().cpu().numpy().copy()

    @property
    def temperature(self) -> float:
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        """Set sampling temperature; zero gives deterministic component means."""
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        self._temperature = float(temperature)

    def set_latent(self, values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        """Replace the current latent vector and return its decoded image.

        The recurrent hidden state is retained, allowing an intervention on
        the current representation without discarding sequence context.
        """
        if self._hidden is None:
            raise RuntimeError("Call reset() before editing the latent state")
        latent_array = np.asarray(values, dtype=np.float32)
        if latent_array.shape != (self.vae.latent_dim,) or not np.isfinite(latent_array).all():
            raise ValueError(
                f"latent values must contain {self.vae.latent_dim} finite numbers"
            )
        self._latent = torch.as_tensor(
            latent_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return self._decode(self._latent)

    def reset(
        self,
        seed: int | None = None,
        episode_index: int | None = None,
        frame_index: int | None = None,
    ) -> np.ndarray:
        """Reset from an encoded real frame and return its decoded RGB image.

        ``episode_index`` selects a sorted file in ``seed_dir``. If omitted, an
        episode is selected randomly. ``frame_index`` optionally selects a
        specific frame within that episode.
        """
        if episode_index is not None and not 0 <= episode_index < len(self._seed_files):
            raise IndexError("episode_index is outside the available seed episodes")
        self._rng = np.random.default_rng(seed)
        selected_episode = (
            episode_index
            if episode_index is not None
            else int(self._rng.integers(0, len(self._seed_files)))
        )
        with np.load(self._seed_files[selected_episode]) as episode:
            latents = episode["latents"]
        if len(latents) == 0:
            raise ValueError(f"Seed episode {self._seed_files[selected_episode]} is empty")
        if frame_index is None:
            frame_index = int(self._rng.integers(0, len(latents)))
        if not 0 <= frame_index < len(latents):
            raise IndexError("frame_index is outside the selected seed episode")

        self._latent = torch.as_tensor(
            latents[frame_index], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._hidden = self.dynamics.initial_state(batch_size=1, device=self.device)
        return self._decode(self._latent)

    def step(self, action: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        """Advance one dream step and return an RGB uint8 image.

        Action order is steering, gas, brake. Values are clipped to the
        CarRacing action range before being passed to the dynamics model.
        """
        if self._latent is None or self._hidden is None:
            raise RuntimeError("Call reset() before step()")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (3,) or not np.isfinite(action_array).all():
            raise ValueError("action must contain three finite values")
        action_array = np.clip(action_array, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        action_tensor = torch.as_tensor(
            action_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            self._latent, self._hidden = self.dynamics.step(
                self._latent,
                action_tensor,
                self._hidden,
                temperature=self._temperature,
            )
            return self._decode(self._latent)

    def _decode(self, latent: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            image = self.vae.decode(latent)[0].permute(1, 2, 0).cpu().numpy()
        return (image * 255.0).round().clip(0, 255).astype(np.uint8)
