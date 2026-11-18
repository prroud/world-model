"""Encode collected episodes into deterministic VAE latent sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from world_model.vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/carracing"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/vae/vae.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/latents"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    source_files = sorted(args.data.glob("episode_*.npz"))
    if not source_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.data}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    existing_files = list(args.output.glob("episode_*.npz"))
    if existing_files:
        raise FileExistsError(
            f"{args.output} already contains latent episode files; choose an empty output directory."
        )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    latent_dim = int(checkpoint["latent_dim"])
    model = VAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    with torch.inference_mode():
        for source_path in source_files:
            with np.load(source_path) as episode:
                observations = episode["observations"]
                actions = episode["actions"].copy()
                dones = episode["dones"].copy()
            latent_batches: list[np.ndarray] = []
            for start in range(0, len(observations), args.batch_size):
                batch = observations[start : start + args.batch_size]
                images = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2)
                images = images.to(device=device, dtype=torch.float32).div_(255.0)
                mu, _ = model.encode(images)
                latent_batches.append(mu.cpu().numpy().astype(np.float32, copy=False))

            output_path = args.output / source_path.name
            with output_path.open("wb") as archive_file:
                np.savez_compressed(
                    archive_file,
                    latents=np.concatenate(latent_batches, axis=0),
                    actions=actions,
                    dones=dones,
                    latent_dim=np.asarray(latent_dim, dtype=np.int32),
                    source_episode=np.asarray(source_path.name),
                )
            print(f"Encoded {len(observations)} frames -> {output_path}")


if __name__ == "__main__":
    main()
