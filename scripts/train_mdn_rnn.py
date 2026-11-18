"""Train the MDN-RNN on encoded CarRacing latent sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from world_model.mdn_rnn import MDNRNN, mixture_nll


def make_windows(
    files: list[Path], sequence_length: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load episodes and split them into non-overlapping recurrent windows."""
    windows = []
    for path in files:
        with np.load(path) as episode:
            latents = episode["latents"]
            actions = episode["actions"]
        if len(latents) != len(actions):
            raise ValueError(f"Latent/action length mismatch in {path}")
        for start in range(0, len(latents) - 1, sequence_length):
            end = start + sequence_length + 1
            if end > len(latents):
                break
            windows.append(
                (latents[start : end - 1], actions[start : end - 1], latents[start + 1 : end])
            )
    return windows


def batches(
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    rng: np.random.Generator,
):
    order = np.arange(len(windows))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [windows[index] for index in order[start : start + batch_size]]
        latent_inputs = np.stack([window[0] for window in selected])
        action_inputs = np.stack([window[1] for window in selected])
        targets = np.stack([window[2] for window in selected])
        yield tuple(
            torch.from_numpy(array).to(device=device, dtype=torch.float32)
            for array in (latent_inputs, action_inputs, targets)
        )


def run_epoch(
    model: MDNRNN,
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = total_steps = 0.0
    with torch.set_grad_enabled(training):
        for latents, actions, targets in batches(
            windows, batch_size, device, training, rng
        ):
            output = model(latents, actions)
            loss = mixture_nll(output, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            count = latents.shape[0] * latents.shape[1]
            total_loss += loss.item() * count
            total_steps += count
    if total_steps == 0:
        raise ValueError("The selected dataset split contains no complete training windows")
    return total_loss / total_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/latents"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mdn_rnn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.sequence_length) < 1:
        parser.error("epochs, batch-size, and sequence-length must be at least 1")
    if not 0 <= args.validation_fraction < 1:
        parser.error("validation fraction must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files = sorted(args.data.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.data}")
    rng.shuffle(files)
    validation_count = (
        max(1, round(len(files) * args.validation_fraction))
        if len(files) > 1 and args.validation_fraction > 0
        else 0
    )
    validation_files = files[:validation_count]
    training_files = files[validation_count:]
    train_windows = make_windows(training_files, args.sequence_length)
    validation_windows = make_windows(validation_files, args.sequence_length)
    if not train_windows:
        raise ValueError("No complete windows found; collect longer episodes or reduce sequence length")
    if validation_files and not validation_windows:
        raise ValueError("Validation episodes are too short for the selected sequence length")

    # Read dimensions from the encoded data rather than assuming the VAE settings.
    with np.load(training_files[0]) as first_episode:
        latent_dim = int(first_episode["latents"].shape[-1])
    action_dim = int(train_windows[0][1].shape[-1])
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    args.output.mkdir(parents=True, exist_ok=True)
    model = MDNRNN(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    history_path = args.output / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history = csv.writer(history_file)
        history.writerow(["epoch", "train_nll", "validation_nll"])
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                model, train_windows, args.batch_size, device, rng, optimizer
            )
            val_loss = (
                run_epoch(model, validation_windows, args.batch_size, device, rng)
                if validation_windows
                else train_loss
            )
            history.writerow([epoch, train_loss, val_loss])
            history_file.flush()
            print(f"Epoch {epoch:03d}: train NLL={train_loss:.4f}, "
                  f"validation NLL={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "latent_dim": latent_dim,
                        "action_dim": action_dim,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "num_mixtures": args.num_mixtures,
                        "epoch": epoch,
                        "validation_nll": val_loss,
                    },
                    args.output / "mdn_rnn.pt",
                )
    print(f"Saved best checkpoint to {args.output / 'mdn_rnn.pt'}")


if __name__ == "__main__":
    main()
