# World Model Demo

An interactive world model demonstrator based on Gymnasium's `CarRacing-v3`. A variational autoencoder (VAE) compresses 64×64 RGB observations into 32-dimensional latent vectors. An MDN-RNN predicts future latent states from the previous state and continuous steering, gas, and brake inputs. The browser demo decodes those predictions into frames.

This is an experimental learned simulation. It can drift from the real environment, especially over long open-loop rollouts. The optional Gymnasium comparison is diagnostic: it applies the same controls to both environments, but does not synchronize their physical states.

## Quick start with trained assets

Use Python 3.10 or newer. From the repository root, install the project and start the web app:

```powershell
python -m pip install -e .
uvicorn world_model.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. This requires the trained VAE and MDN-RNN checkpoints plus encoded latent seed episodes at the default paths below. The `/health` endpoint reports whether those files are present.

Use the arrow keys or WASD to drive. The page provides temperature control, latent-state inspection and editing, selectable reset seeds, uncertainty telemetry, and an optional side-by-side Gymnasium comparison. `Record session` saves a browser WebM recording and a JSON control log; stochastic sampling means the log is not a pixel-identical replay.

## Full pipeline

Install the project dependencies once:

```powershell
python -m pip install -e .
```

Then collect episodes, validate the dataset, train both models, encode latent sequences, and start the app:

```powershell
python scripts/run_demo.py --episodes 20 --vae-epochs 20 --rnn-epochs 30
```

The first run can take a while, particularly on CPU. The launcher refuses to overwrite its expected dataset and model outputs. Choose a clean workspace or move existing outputs before starting another full build. Use `--device cuda` when a supported CUDA setup is available, `--render` to show collection, or `--host` and `--port` to change the server address.

### Run stages separately

Each command below runs from the repository root. These are also useful when you want to inspect or repeat a single stage.

```powershell
# Collect episodes as compressed NumPy archives
python scripts/collect_data.py --episodes 20 --output data/carracing

# Check shapes, ranges, episode boundaries, and image samples
python scripts/validate_dataset.py --data data/carracing --json-out artifacts/data-audit.json --preview-out artifacts/data-preview.png

# Train the image encoder and decoder; save the best checkpoint and history
python scripts/train_vae.py --data data/carracing --epochs 20

# Encode each episode using the VAE posterior mean
python scripts/encode_latents.py --data data/carracing --checkpoint artifacts/vae/vae.pt --output data/latents

# Train the recurrent dynamics model
python scripts/train_mdn_rnn.py --data data/latents --epochs 30
```

Collection writes one `episode_*.npz` file per episode with observations, actions, rewards, done flags, and seed metadata. The dataset validator reports errors and warnings and writes a contact sheet for visual inspection. Training reserves complete episodes for validation; the VAE writes reconstruction examples and both trainers save per-epoch CSV histories.

## Evaluate and report

Evaluate open-loop predictions on held-out recorded actions. For example, compare three episodes at several horizons:

```powershell
python scripts/evaluate_rollout.py --episode-indices 0,1,2 --horizons 25,50,100 --temperature 0
```

The rollout evaluation saves real-versus-generated contact sheets, per-step pixel-error CSVs, and `artifacts/rollout/summary.csv`. Measure inference throughput separately:

```powershell
python scripts/benchmark_engine.py --steps 500 --json-out artifacts/benchmark.json
```

Generate a Markdown report, a machine-readable JSON summary, and SVG charts from whichever pipeline results are available:

```powershell
python scripts/generate_report.py
```

The report is written to `artifacts/report/report.md`; `summary.json` and any available charts are saved alongside it. It does not retrain models, and missing inputs are called out in the report. By default it reads `artifacts/data-audit.json`, model histories, the rollout summary, and the benchmark. Use `--artifacts-dir` to read another artifact directory or `--output-dir` to choose a different report destination.

## Model and engine interface

The VAE implementation is in `src/world_model/vae.py`; the MDN-RNN is in `src/world_model/mdn_rnn.py`. `src/world_model/engine.py` combines their checkpoints for frame-by-frame generation. It starts from an encoded episode seed and then advances without stepping Gymnasium:

```python
from world_model.engine import WorldEngine

engine = WorldEngine()
frame = engine.reset(seed=7)
frame = engine.step([0.0, 0.8, 0.0])  # steering, gas, brake
engine.set_temperature(0.5)
```

Each frame is an RGB `uint8` array with shape `(64, 64, 3)`. The API is implemented in `src/world_model/api.py`; its WebSocket endpoint is `/ws/dream`, and the browser assets are in `src/world_model/static/`. The browser's `Model evaluation` panel reads saved training histories, rollout summaries, and benchmark metrics through `/metrics`.

## Repository layout

```text
src/world_model/       Models, engine, API, and browser application
scripts/               Data collection, training, evaluation, and reporting
tests/                 Unit tests using synthetic inputs
data/                  Collected episodes and encoded latent sequences (local)
artifacts/              Checkpoints, metrics, visualizations, and reports (local)
```

Datasets, model checkpoints, and generated reports are local artifacts and are not tracked in Git.

## Tests and CI

Install the test dependencies and run the unit suite with:

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

GitHub Actions runs the suite on Python 3.10 and 3.12. The tests use synthetic inputs and do not need collected episodes or trained checkpoints.
