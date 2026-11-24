"""Compare an open-loop dream rollout with a held-out recorded episode."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from world_model.mdn_rnn import MDNRNN
from world_model.vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-data", type=Path, default=Path("data/latents"))
    parser.add_argument("--image-data", type=Path, default=Path("data/carracing"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("artifacts/vae/vae.pt"))
    parser.add_argument("--dynamics-checkpoint", type=Path, default=Path("artifacts/mdn_rnn/mdn_rnn.pt"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--episode-indices",
        help="Comma-separated episode indices; overrides --episode-index",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--horizons",
        help="Comma-separated rollout horizons; overrides --horizon",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rollout"))
    args = parser.parse_args()
    try:
        args.selected_episodes = (
            [int(value) for value in args.episode_indices.split(",")]
            if args.episode_indices
            else [args.episode_index]
        )
        args.selected_horizons = (
            [int(value) for value in args.horizons.split(",")]
            if args.horizons
            else [args.horizon]
        )
    except ValueError:
        parser.error("episode-indices and horizons must be comma-separated integers")
    if (
        not args.selected_episodes
        or not args.selected_horizons
        or
        min(args.selected_episodes) < 0
        or args.start_frame < 0
        or min(args.selected_horizons) < 1
        or len(set(args.selected_episodes)) != len(args.selected_episodes)
        or len(set(args.selected_horizons)) != len(args.selected_horizons)
    ):
        parser.error(
            "episode indices and start-frame must be nonnegative; horizons must be positive and lists nonempty and unique"
        )
    if not np.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    return args


def decode_image(vae: VAE, latent: torch.Tensor) -> np.ndarray:
    image = vae.decode(latent)[0].permute(1, 2, 0).cpu().numpy()
    return (image * 255.0).round().clip(0, 255).astype(np.uint8)


def save_contact_sheet(
    real_frames: list[np.ndarray],
    dream_frames: list[np.ndarray],
    output_path: Path,
    columns: int = 8,
) -> None:
    tile_width, tile_height = 128, 78
    rows = math.ceil(len(real_frames) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#12161a")
    draw = ImageDraw.Draw(sheet)
    for index, (real, dream) in enumerate(zip(real_frames, dream_frames)):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        draw.text((x + 3, y + 1), f"{index + 1:03d} REAL", fill="#c8f169")
        draw.text((x + 67, y + 1), "DREAM", fill="#86afd0")
        sheet.paste(Image.fromarray(real), (x, y + 13))
        sheet.paste(Image.fromarray(dream), (x + 64, y + 13))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    latent_files = sorted(args.latent_data.glob("episode_*.npz"))
    if not latent_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.latent_data}")
    if max(args.selected_episodes) >= len(latent_files):
        raise IndexError(f"episode indices must be below {len(latent_files)}")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    vae_payload = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=True)
    dynamics_payload = torch.load(args.dynamics_checkpoint, map_location="cpu", weights_only=True)

    vae = VAE(latent_dim=int(vae_payload["latent_dim"]))
    vae.load_state_dict(vae_payload["state_dict"])
    vae.to(device).eval()
    dynamics = MDNRNN(
        latent_dim=int(dynamics_payload["latent_dim"]),
        action_dim=int(dynamics_payload["action_dim"]),
        hidden_size=int(dynamics_payload["hidden_size"]),
        num_layers=int(dynamics_payload["num_layers"]),
        num_mixtures=int(dynamics_payload["num_mixtures"]),
    )
    if dynamics.latent_dim != vae.latent_dim:
        raise ValueError("VAE and MDN-RNN checkpoints use different latent dimensions")
    dynamics.load_state_dict(dynamics_payload["state_dict"])
    dynamics.to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for episode_index in args.selected_episodes:
        latent_path = latent_files[episode_index]
        with np.load(latent_path) as episode:
            latents = episode["latents"]
            actions = episode["actions"]
            source_name = str(episode["source_episode"].item())
        source_path = args.image_data / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image episode not found: {source_path}")
        with np.load(source_path) as episode:
            observations = episode["observations"]
        available_steps = (
            min(len(latents), len(actions), len(observations)) - args.start_frame - 1
        )
        if available_steps < max(args.selected_horizons):
            raise ValueError(
                f"Episode {episode_index} has {max(0, available_steps)} steps available; "
                f"requested {max(args.selected_horizons)}"
            )

        latent = torch.as_tensor(
            latents[args.start_frame], dtype=torch.float32, device=device
        ).unsqueeze(0)
        hidden = dynamics.initial_state(batch_size=1, device=device)
        real_frames: list[np.ndarray] = []
        dream_frames: list[np.ndarray] = []
        errors: list[float] = []
        with torch.inference_mode():
            for offset in range(max(args.selected_horizons)):
                action = torch.as_tensor(
                    actions[args.start_frame + offset], dtype=torch.float32, device=device
                ).unsqueeze(0)
                latent, hidden = dynamics.step(latent, action, hidden, args.temperature)
                dream = decode_image(vae, latent)
                real = observations[args.start_frame + offset + 1]
                real_frames.append(real)
                dream_frames.append(dream)
                errors.append(float(np.abs(real.astype(np.float32) - dream).mean()))

        for horizon in args.selected_horizons:
            selected = errors[:horizon]
            window = min(10, horizon)
            summary_rows.append(
                [
                    episode_index,
                    source_name,
                    horizon,
                    float(np.mean(selected)),
                    float(np.mean(selected[:window])),
                    float(np.mean(selected[-window:])),
                ]
            )
        image_path = args.output / f"episode_{episode_index:05d}_rollout.png"
        save_contact_sheet(real_frames, dream_frames, image_path)
        metrics_path = args.output / f"episode_{episode_index:05d}_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(["step", "pixel_mae_0_255"])
            writer.writerows((index + 1, error) for index, error in enumerate(errors))
        print(f"Episode {episode_index} ({source_name}): saved {image_path} and {metrics_path}")

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            [
                "episode_index",
                "source_episode",
                "horizon",
                "mean_mae_0_255",
                "first_window_mae_0_255",
                "last_window_mae_0_255",
            ]
        )
        writer.writerows(summary_rows)
    print(f"Saved {len(summary_rows)} episode/horizon summaries to {summary_path}")


if __name__ == "__main__":
    main()
