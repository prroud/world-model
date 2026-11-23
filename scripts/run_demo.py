"""Build the demo assets from scratch, then serve the interactive web app."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--vae-epochs", type=int, default=20)
    parser.add_argument("--rnn-epochs", type=int, default=30)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--render", action="store_true", help="Show CarRacing while collecting data")
    args = parser.parse_args()
    if args.episodes < 1 or args.vae_epochs < 1 or args.rnn_epochs < 1:
        parser.error("episodes and epoch counts must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    return args


def run_stage(label: str, *command: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_fresh_outputs() -> None:
    protected = [
        ROOT / "data/carracing/episode_*.npz",
        ROOT / "data/latents/episode_*.npz",
        ROOT / "artifacts/data-audit.json",
        ROOT / "artifacts/data-preview.png",
        ROOT / "artifacts/vae/vae.pt",
        ROOT / "artifacts/vae/history.csv",
        ROOT / "artifacts/vae/reconstructions.png",
        ROOT / "artifacts/mdn_rnn/mdn_rnn.pt",
        ROOT / "artifacts/mdn_rnn/history.csv",
    ]
    existing = [
        str(path.relative_to(ROOT))
        for pattern in protected
        for path in pattern.parent.glob(pattern.name)
    ]
    if existing:
        paths = "\n  ".join(existing[:10])
        more = "\n  …" if len(existing) > 10 else ""
        raise FileExistsError(
            "The full demo build writes new dataset and model artifacts. "
            f"These outputs already exist:\n  {paths}{more}\n"
            "Move them aside or remove them before starting a fresh build."
        )


def main() -> None:
    args = parse_args()
    ensure_fresh_outputs()
    python = sys.executable
    device = ["--device", args.device]

    collect = [python, "scripts/collect_data.py", "--episodes", str(args.episodes)]
    if args.render:
        collect.append("--render")
    run_stage("Collecting trajectories", *collect)
    run_stage(
        "Auditing trajectories",
        python,
        "scripts/validate_dataset.py",
        "--data",
        "data/carracing",
        "--json-out",
        "artifacts/data-audit.json",
        "--preview-out",
        "artifacts/data-preview.png",
    )
    run_stage(
        "Training VAE",
        python,
        "scripts/train_vae.py",
        "--data",
        "data/carracing",
        "--epochs",
        str(args.vae_epochs),
        *device,
    )
    run_stage(
        "Encoding latent episodes",
        python,
        "scripts/encode_latents.py",
        "--data",
        "data/carracing",
        "--checkpoint",
        "artifacts/vae/vae.pt",
        "--output",
        "data/latents",
        *device,
    )
    run_stage(
        "Training MDN-RNN",
        python,
        "scripts/train_mdn_rnn.py",
        "--data",
        "data/latents",
        "--epochs",
        str(args.rnn_epochs),
        *device,
    )

    print(
        f"\nDemo assets are ready. Open http://{args.host}:{args.port}/ "
        "(Ctrl+C stops the server).",
        flush=True,
    )
    run_stage(
        "Starting web demo",
        python,
        "-m",
        "uvicorn",
        "world_model.api:app",
        "--app-dir",
        "src",
        "--host",
        args.host,
        "--port",
        str(args.port),
    )


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
