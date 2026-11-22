"""Train the convolutional VAE on collected CarRacing episodes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from world_model.vae import VAE


def episode_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {data_dir}")
    return files


def episode_batches(
    files: list[Path], batch_size: int, device: torch.device, shuffle: bool,
    rng: np.random.Generator,
):
    """Yield frame batches while keeping at most one episode in memory."""
    ordered_files = list(files)
    if shuffle:
        rng.shuffle(ordered_files)
    for path in ordered_files:
        with np.load(path) as episode:
            observations = episode["observations"]
        indices = np.arange(len(observations))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = observations[indices[start : start + batch_size]]
            tensor = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2)
            yield tensor.to(device=device, dtype=torch.float32).div_(255.0)


def vae_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    foreground_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ELBO components with extra weight on non-background pixels."""
    background_color = images.flatten(2).median(dim=2).values[:, :, None, None]
    color_distance = (images - background_color).square().mean(dim=1, keepdim=True).sqrt()
    pixel_weights = 1.0 + foreground_weight * color_distance
    pixel_weights = pixel_weights / pixel_weights.mean(dim=(2, 3), keepdim=True)
    reconstruction_per_sample = (
        (reconstruction - images).square() * pixel_weights
    ).flatten(1).sum(dim=1)
    kl_per_sample = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
    reconstruction_loss = reconstruction_per_sample.mean()
    kl_loss = kl_per_sample.mean()
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def run_epoch(
    model: VAE,
    files: list[Path],
    batch_size: int,
    device: torch.device,
    beta: float,
    foreground_weight: float,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_reconstruction = total_kl = total_samples = 0.0
    with torch.set_grad_enabled(training):
        for images in episode_batches(files, batch_size, device, training, rng):
            reconstruction, mu, logvar = model(images, sample=training)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction, images, mu, logvar, beta, foreground_weight
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = len(images)
            total_loss += loss.item() * count
            total_reconstruction += reconstruction_loss.item() * count
            total_kl += kl_loss.item() * count
            total_samples += count
    if total_samples == 0:
        raise ValueError("The selected dataset split contains no frames")
    return (
        total_loss / total_samples,
        total_reconstruction / total_samples,
        total_kl / total_samples,
    )


def save_reconstruction_preview(
    model: VAE, files: list[Path], output_path: Path, device: torch.device
) -> None:
    with np.load(files[0]) as episode:
        frames = episode["observations"][:8]
    images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    images = images.to(device=device, dtype=torch.float32).div_(255.0)
    model.eval()
    with torch.inference_mode():
        reconstructions, _, _ = model(images, sample=False)
    originals = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    decoded = (
        reconstructions.cpu().permute(0, 2, 3, 1).numpy() * 255
    ).round().clip(0, 255).astype(np.uint8)
    rows = [np.concatenate((originals[i], decoded[i]), axis=1) for i in range(len(frames))]
    Image.fromarray(np.concatenate(rows, axis=0)).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/carracing"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vae"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--foreground-weight", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.kl_warmup_epochs < 0:
        parser.error("--epochs and --batch-size must be positive; KL warmup cannot be negative")
    if args.beta < 0 or args.foreground_weight < 0 or not 0 <= args.validation_fraction < 1:
        parser.error("beta and foreground weight must be nonnegative; validation fraction in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    files = episode_files(args.data)
    rng.shuffle(files)
    validation_count = (
        max(1, round(len(files) * args.validation_fraction))
        if len(files) > 1 and args.validation_fraction > 0
        else 0
    )
    validation_files = files[:validation_count]
    training_files = files[validation_count:]
    if not training_files:
        raise ValueError("At least one episode must remain for training")

    args.output.mkdir(parents=True, exist_ok=True)
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_selection_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    print(f"Training on {device}; {len(training_files)} train episodes, "
          f"{len(validation_files)} validation episodes")

    with (args.output / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as history_file:
        history = csv.writer(history_file)
        history.writerow(
            [
                "epoch",
                "beta_used",
                "train_loss",
                "train_reconstruction",
                "train_kl",
                "validation_loss",
                "validation_reconstruction",
                "validation_kl",
            ]
        )
        for epoch in range(1, args.epochs + 1):
            beta_used = args.beta * min(
                1.0,
                epoch / args.kl_warmup_epochs if args.kl_warmup_epochs else 1.0,
            )
            train_loss, train_recon, train_kl = run_epoch(
                model, training_files, args.batch_size, device, beta_used,
                args.foreground_weight, rng, optimizer
            )
            if validation_files:
                val_loss, val_recon, val_kl = run_epoch(
                    model, validation_files, args.batch_size, device, beta_used,
                    args.foreground_weight, rng
                )
                selection_loss = val_recon + args.beta * val_kl
                history.writerow(
                    [epoch, beta_used, train_loss, train_recon, train_kl,
                     val_loss, val_recon, val_kl]
                )
                print(f"Epoch {epoch:03d} (beta={beta_used:.4f}): train={train_loss:.2f} "
                      f"(recon={train_recon:.2f}, kl={train_kl:.2f}) "
                      f"val={val_loss:.2f} (recon={val_recon:.2f}, kl={val_kl:.2f})")
            else:
                selection_loss = train_recon + args.beta * train_kl
                history.writerow(
                    [epoch, beta_used, train_loss, train_recon, train_kl, "", "", ""]
                )
                print(f"Epoch {epoch:03d} (beta={beta_used:.4f}): train={train_loss:.2f} "
                      f"(recon={train_recon:.2f}, kl={train_kl:.2f})")
            history_file.flush()
            if selection_loss < best_selection_loss:
                best_selection_loss = selection_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "state_dict": best_state,
                        "latent_dim": args.latent_dim,
                        "epoch": epoch,
                        "selection_loss": selection_loss,
                        "beta": args.beta,
                        "kl_warmup_epochs": args.kl_warmup_epochs,
                        "foreground_weight": args.foreground_weight,
                    },
                    args.output / "vae.pt",
                )

    if best_state is not None:
        model.load_state_dict(best_state)
    save_reconstruction_preview(model, files, args.output / "reconstructions.png", device)
    print(f"Saved checkpoint, history, and preview to {args.output}")


if __name__ == "__main__":
    main()
