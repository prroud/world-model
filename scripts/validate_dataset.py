"""Audit collected CarRacing episode archives before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ACTION_NAMES = ("steering", "gas", "brake")


def action_statistics(actions: np.ndarray) -> dict | None:
    if (
        actions.ndim != 2
        or actions.shape[1:] != (3,)
        or not np.issubdtype(actions.dtype, np.number)
        or np.issubdtype(actions.dtype, np.complexfloating)
        or not np.isfinite(actions).all()
        or len(actions) == 0
    ):
        return None
    return {
        name: {
            "min": float(actions[:, index].min()),
            "max": float(actions[:, index].max()),
            "mean": float(actions[:, index].mean()),
            "std": float(actions[:, index].std()),
        }
        for index, name in enumerate(ACTION_NAMES)
    }


def scalar_statistics(values: np.ndarray) -> dict | None:
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or not np.isfinite(values).all()
        or len(values) == 0
    ):
        return None
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def save_preview(paths: list[Path], output_path: Path, samples_per_episode: int) -> None:
    tiles: list[tuple[str, np.ndarray]] = []
    for path in paths:
        try:
            with np.load(path) as archive:
                frames = archive["observations"]
            if frames.ndim != 4 or frames.shape[1:] != (64, 64, 3) or len(frames) == 0:
                continue
            indices = np.linspace(0, len(frames) - 1, min(samples_per_episode, len(frames)), dtype=int)
            tiles.extend((f"{path.stem} · {index}", frames[index]) for index in indices)
        except Exception:
            continue
    if not tiles:
        raise ValueError("No valid observation frames available for preview")

    columns = min(4, len(tiles))
    tile_width, tile_height = 160, 84
    sheet = Image.new("RGB", (columns * tile_width, ((len(tiles) + columns - 1) // columns) * tile_height), "#12161a")
    draw = ImageDraw.Draw(sheet)
    for position, (label, frame) in enumerate(tiles):
        x = (position % columns) * tile_width
        y = (position // columns) * tile_height
        draw.text((x + 3, y + 2), label, fill="white")
        sheet.paste(Image.fromarray(frame), (x, y + 18))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def inspect_episode(path: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        with np.load(path) as episode:
            required = {"observations", "actions", "rewards", "dones"}
            missing = sorted(required.difference(episode.files))
            if missing:
                return {"file": path.name, "errors": [f"missing arrays: {', '.join(missing)}"], "warnings": []}
            observations = episode["observations"]
            actions = episode["actions"]
            rewards = episode["rewards"]
            dones = episode["dones"]
            episode_index = episode["episode_index"].item() if "episode_index" in episode else None
            episode_seed = episode["episode_seed"].item() if "episode_seed" in episode else None
    except Exception as exc:
        return {"file": path.name, "errors": [f"cannot read archive: {exc}"], "warnings": []}

    if episode_index is None or episode_seed is None:
        warnings.append("missing episode seed/index metadata; exact replay may not be possible")
    elif not isinstance(episode_index, (int, np.integer)) or not isinstance(
        episode_seed, (int, np.integer)
    ):
        warnings.append("episode seed/index metadata must be integer scalars")

    if observations.ndim != 4 or observations.shape[1:] != (64, 64, 3):
        errors.append(f"observations must have shape (T, 64, 64, 3), got {observations.shape}")
    elif observations.dtype != np.uint8:
        errors.append(f"observations must use uint8, got {observations.dtype}")
    if actions.ndim != 2 or actions.shape[1:] != (3,):
        errors.append(f"actions must have shape (T, 3), got {actions.shape}")
    if not np.issubdtype(actions.dtype, np.number) or np.issubdtype(actions.dtype, np.complexfloating):
        errors.append(f"actions must use a real numeric dtype, got {actions.dtype}")
    elif not np.isfinite(actions).all():
        errors.append("actions contain non-finite values")
    elif actions.ndim == 2 and actions.shape[1:] == (3,):
        low = np.asarray([-1.0, 0.0, 0.0])
        high = np.asarray([1.0, 1.0, 1.0])
        if np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
            errors.append("actions are outside the CarRacing action range")
    if rewards.ndim != 1:
        errors.append(f"rewards must have shape (T,), got {rewards.shape}")
    if not np.issubdtype(rewards.dtype, np.number) or np.issubdtype(rewards.dtype, np.complexfloating):
        errors.append(f"rewards must use a real numeric dtype, got {rewards.dtype}")
    elif not np.isfinite(rewards).all():
        errors.append("rewards contain non-finite values")
    if dones.ndim != 1:
        errors.append(f"dones must have shape (T,), got {dones.shape}")
    if dones.dtype != np.bool_:
        errors.append(f"dones must use bool, got {dones.dtype}")

    lengths = [array.shape[0] for array in (observations, actions, rewards, dones) if array.ndim > 0]
    if len(lengths) != 4 or len(set(lengths)) != 1:
        errors.append("observations, actions, rewards, and dones have different lengths")
    observation_count = observations.shape[0] if observations.ndim > 0 else 0
    if not observation_count:
        errors.append("episode is empty")
    elif dones.ndim == 1 and len(dones) == observation_count and dones.dtype == np.bool_:
        if not dones[-1] or np.any(dones[:-1]):
            errors.append("dones must be false except for a true final entry")

    if observations.ndim == 4 and observations.shape[1:] == (64, 64, 3) and observation_count:
        blank_count = int(np.all(observations == 0, axis=(1, 2, 3)).sum())
        if blank_count:
            warnings.append(f"{blank_count} fully black observation(s)")
        if len(observations) > 1:
            duplicate_count = int(np.all(observations[1:] == observations[:-1], axis=(1, 2, 3)).sum())
            if duplicate_count:
                warnings.append(f"{duplicate_count} consecutive duplicate frame(s)")

    return {
        "file": path.name,
        "frames": int(len(observations)),
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "episode_return": (
            float(rewards.sum())
            if rewards.ndim == 1
            and np.issubdtype(rewards.dtype, np.number)
            and not np.issubdtype(rewards.dtype, np.complexfloating)
            and np.isfinite(rewards).all()
            else None
        ),
        "mean_step_reward": (
            float(rewards.mean())
            if rewards.ndim == 1
            and len(rewards)
            and np.issubdtype(rewards.dtype, np.number)
            and not np.issubdtype(rewards.dtype, np.complexfloating)
            and np.isfinite(rewards).all()
            else None
        ),
        "action_statistics": action_statistics(actions),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/carracing"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--preview-out", type=Path, help="Write a contact sheet of evenly sampled frames")
    parser.add_argument("--samples-per-episode", type=int, default=4)
    args = parser.parse_args()

    if args.samples_per_episode < 1:
        parser.error("--samples-per-episode must be at least 1")

    paths = sorted(args.data.glob("episode_*.npz"))
    if not paths:
        parser.error(f"No episode_*.npz files found in {args.data}")
    episodes = [inspect_episode(path) for path in paths]
    all_actions = []
    for path in paths:
        try:
            with np.load(path) as archive:
                actions = archive["actions"]
            if action_statistics(actions) is not None:
                all_actions.append(actions.astype(np.float64, copy=False))
        except Exception:
            continue
    combined_actions = np.concatenate(all_actions) if all_actions else np.empty((0, 3))
    valid_returns = np.asarray(
        [episode["episode_return"] for episode in episodes if episode.get("episode_return") is not None],
        dtype=np.float64,
    )
    episode_lengths = np.asarray(
        [episode["frames"] for episode in episodes if episode.get("frames", 0) > 0],
        dtype=np.float64,
    )
    report = {
        "directory": str(args.data),
        "episode_count": len(episodes),
        "frame_count": sum(episode.get("frames", 0) for episode in episodes),
        "error_count": sum(len(episode["errors"]) for episode in episodes),
        "warning_count": sum(len(episode["warnings"]) for episode in episodes),
        "action_statistics": action_statistics(combined_actions),
        "episode_length_statistics": scalar_statistics(episode_lengths),
        "episode_return_statistics": scalar_statistics(valid_returns),
        "episodes": episodes,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved audit report to {args.json_out}")
    if args.preview_out:
        save_preview(paths, args.preview_out, args.samples_per_episode)
        print(f"Saved frame preview to {args.preview_out}")
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
