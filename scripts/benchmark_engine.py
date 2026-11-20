"""Measure end-to-end WorldEngine frame latency and throughput."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from world_model.engine import WorldEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("artifacts/vae/vae.pt"))
    parser.add_argument("--dynamics-checkpoint", type=Path, default=Path("artifacts/mdn_rnn/mdn_rnn.pt"))
    parser.add_argument("--seed-data", type=Path, default=Path("data/latents"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup cannot be negative")
    if not np.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    return args


def main() -> None:
    args = parse_args()
    engine = WorldEngine(
        vae_checkpoint=args.vae_checkpoint,
        dynamics_checkpoint=args.dynamics_checkpoint,
        seed_dir=args.seed_data,
        device=args.device,
        temperature=args.temperature,
    )
    engine.reset(seed=args.seed)
    actions_rng = np.random.default_rng(args.seed)
    actions = [
        np.asarray(
            [actions_rng.uniform(-1.0, 1.0), actions_rng.uniform(0.6, 1.0), 0.0],
            dtype=np.float32,
        )
        for _ in range(args.steps)
    ]

    for index in range(args.warmup):
        engine.step(actions[index % len(actions)])

    latencies_ms = []
    for action in actions:
        started = time.perf_counter()
        engine.step(action)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    mean_latency = float(latency_array.mean())
    hardware = platform.processor() or platform.machine()
    if engine.device.type == "cuda":
        hardware = torch.cuda.get_device_name(engine.device)
    report = {
        "steps": args.steps,
        "warmup_steps": args.warmup,
        "temperature": args.temperature,
        "device": str(engine.device),
        "hardware": hardware,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.percentile(latency_array, 50)),
        "p95_latency_ms": float(np.percentile(latency_array, 95)),
        "p99_latency_ms": float(np.percentile(latency_array, 99)),
        "frames_per_second": 1000.0 / mean_latency,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved benchmark report to {args.json_out}")


if __name__ == "__main__":
    main()
