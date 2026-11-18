"""Collect image/action trajectories from Gymnasium's CarRacing-v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image


IMAGE_SIZE = (64, 64)


def resize_observation(observation: np.ndarray) -> np.ndarray:
    """Resize an RGB observation and keep it as compact uint8 pixels."""
    image = Image.fromarray(observation)
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def exploratory_action(
    rng: np.random.Generator,
    steering: float,
    steering_target: float,
    step: int,
) -> tuple[np.ndarray, float, float]:
    """Sample a temporally correlated driving action and next steering state."""
    if step % 24 == 0:
        steering_target = rng.uniform(-1.0, 1.0)
    steering = float(np.clip(0.82 * steering + 0.18 * steering_target, -1.0, 1.0))

    gas = float(rng.uniform(0.55, 1.0))
    brake = float(rng.uniform(0.2, 0.7)) if rng.random() < 0.025 else 0.0
    action = np.asarray([steering, gas, brake], dtype=np.float32)
    return action, steering, steering_target


def collect_episodes(
    output_dir: Path,
    episodes: int,
    seed: int,
    render: bool = False,
) -> None:
    """Collect episodes into separate NPZ files to bound memory use."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = list(output_dir.glob("episode_*.npz"))
    if existing_files:
        raise FileExistsError(
            f"{output_dir} already contains episode files; choose an empty output directory."
        )

    env = gym.make("CarRacing-v3", render_mode="human" if render else None)
    rng = np.random.default_rng(seed)
    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            frames: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            steering = 0.0
            steering_target = 0.0
            done = False
            step = 0

            while not done:
                action, steering, steering_target = exploratory_action(
                    rng, steering, steering_target, step
                )
                frames.append(resize_observation(observation))
                actions.append(action)
                observation, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                done = terminated or truncated
                step += 1

            dones = np.zeros(step, dtype=np.bool_)
            dones[-1] = True
            episode_path = output_dir / f"episode_{episode_index:05d}.npz"
            with episode_path.open("wb") as archive_file:
                np.savez_compressed(
                    archive_file,
                    observations=np.stack(frames),
                    actions=np.stack(actions),
                    rewards=np.asarray(rewards, dtype=np.float32),
                    dones=dones,
                    image_size=np.asarray(IMAGE_SIZE, dtype=np.int32),
                    env_id=np.asarray("CarRacing-v3"),
                )
            print(f"Episode {episode_index + 1}/{episodes}: {step} frames -> {episode_path}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/carracing"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--render", action="store_true", help="Show the Gymnasium window")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    collect_episodes(args.output, args.episodes, args.seed, args.render)


if __name__ == "__main__":
    main()
