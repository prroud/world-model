"""Build a human-readable project report from saved training and evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path) -> list[dict[str, str]] | None:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError:
        return None


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def summarize(artifacts: Path) -> dict[str, Any]:
    vae_rows = read_csv(artifacts / "vae/history.csv")
    mdn_rows = read_csv(artifacts / "mdn_rnn/history.csv")
    rollout_rows = read_csv(artifacts / "rollout/summary.csv")
    vae_best = None
    if vae_rows:
        candidates = [
            {key: number(row.get(key)) for key in ("epoch", "validation_reconstruction", "validation_kl", "validation_loss")}
            for row in vae_rows
        ]
        candidates = [row for row in candidates if row["validation_reconstruction"] is not None]
        if candidates:
            vae_best = min(candidates, key=lambda row: row["validation_reconstruction"])
    mdn_best = None
    if mdn_rows:
        candidates = [
            {key: number(row.get(key)) for key in ("epoch", "validation_nll", "train_nll")}
            for row in mdn_rows
        ]
        candidates = [row for row in candidates if row["validation_nll"] is not None]
        if candidates:
            mdn_best = min(candidates, key=lambda row: row["validation_nll"])
    rollout = []
    if rollout_rows:
        by_horizon: dict[int, list[float]] = {}
        for row in rollout_rows:
            horizon = number(row.get("horizon"))
            mae = number(row.get("mean_mae_0_255"))
            if horizon is not None and mae is not None:
                by_horizon.setdefault(int(horizon), []).append(mae)
        rollout = [
            {"horizon": horizon, "episode_count": len(values), "mean_mae_0_255": sum(values) / len(values)}
            for horizon, values in sorted(by_horizon.items())
        ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": read_json(artifacts / "data-audit.json"),
        "vae": {"epochs_recorded": len(vae_rows or []), "best_validation": vae_best},
        "mdn_rnn": {"epochs_recorded": len(mdn_rows or []), "best_validation": mdn_best},
        "rollout": {"episode_horizon_count": len(rollout_rows or []), "by_horizon": rollout},
        "benchmark": read_json(artifacts / "benchmark.json"),
    }


def fmt(value: Any, digits: int = 4) -> str:
    parsed = number(value)
    return f"{parsed:.{digits}f}" if parsed is not None else "—"


def markdown_report(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    vae = summary["vae"]
    mdn = summary["mdn_rnn"]
    benchmark = summary["benchmark"]
    lines = [
        "# World Model Experiment Report",
        "",
        f"Generated at: `{summary['generated_at_utc']}`",
        "",
        "This report summarizes the artifacts currently present in the selected artifacts directory. "
        "The individual files may come from different runs; the report does not infer run identity.",
        "",
        "## Dataset",
    ]
    if dataset:
        lines.extend([
            f"- Episodes: {dataset.get('episode_count', '—')}",
            f"- Frames: {dataset.get('frame_count', '—')}",
            f"- Validation errors: {dataset.get('error_count', '—')}",
            f"- Warnings: {dataset.get('warning_count', '—')}",
        ])
        length = dataset.get("episode_length_statistics") or {}
        returns = dataset.get("episode_return_statistics") or {}
        lines.extend([
            f"- Mean episode length: {fmt(length.get('mean'), 1)} frames",
            f"- Mean episode return: {fmt(returns.get('mean'), 2)}",
        ])
    else:
        lines.append("No dataset audit found (`data-audit.json`).")
    lines.extend(["", "## Model training"])
    if vae["best_validation"]:
        best = vae["best_validation"]
        lines.extend([
            f"- VAE epochs recorded: {vae['epochs_recorded']}",
            f"- Best validation reconstruction loss: {fmt(best['validation_reconstruction'])} (epoch {fmt(best['epoch'], 0)})",
            f"- Validation KL at that epoch: {fmt(best['validation_kl'])}",
        ])
    else:
        lines.append("VAE history is missing or has no validation reconstruction values.")
    if mdn["best_validation"]:
        best = mdn["best_validation"]
        lines.extend([
            f"- MDN-RNN epochs recorded: {mdn['epochs_recorded']}",
            f"- Best validation NLL: {fmt(best['validation_nll'])} (epoch {fmt(best['epoch'], 0)})",
        ])
    else:
        lines.append("MDN-RNN history is missing or has no validation NLL values.")
    lines.extend(["", "## Open-loop evaluation"])
    if summary["rollout"]["by_horizon"]:
        lines.extend([
            "| Horizon (frames) | Episodes | Mean pixel MAE (0–255) |",
            "|---:|---:|---:|",
        ])
        for row in summary["rollout"]["by_horizon"]:
            lines.append(f"| {row['horizon']} | {row['episode_count']} | {fmt(row['mean_mae_0_255'], 2)} |")
    else:
        lines.append("No rollout summary found (`rollout/summary.csv`).")
    lines.extend(["", "## Inference benchmark"])
    if benchmark:
        lines.extend([
            f"- Device: {benchmark.get('device', '—')}; hardware: {benchmark.get('hardware', '—')}",
            f"- Mean latency: {fmt(benchmark.get('mean_latency_ms'), 2)} ms",
            f"- p95 latency: {fmt(benchmark.get('p95_latency_ms'), 2)} ms",
            f"- Throughput: {fmt(benchmark.get('frames_per_second'), 1)} FPS",
        ])
    else:
        lines.append("No inference benchmark found (`benchmark.json`).")
    lines.extend(["", "## Source artifacts", "", "- `data-audit.json`", "- `vae/history.csv`", "- `mdn_rnn/history.csv`", "- `rollout/summary.csv`", "- `benchmark.json`", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    args = parser.parse_args()
    summary = summarize(args.artifacts_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(f"Saved report to {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
