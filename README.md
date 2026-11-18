# World Model Demo

A world model demonstrator for Gymnasium's `CarRacing-v3` environment. The goal
is to generate short, interactive image sequences from user controls using a VAE
and a recurrent MDN-RNN dynamics model.

## Initial scope

- collect 64×64 frames with actions and episode boundaries,
- train a VAE that encodes frames into 32-dimensional latent vectors,
- train an MDN-RNN to predict the next latent state,
- provide a simple web demo with controls, reset, and temperature adjustment,
- compare against Gymnasium diagnostically and measure inference performance.

Manual latent-space editing and long, stable sequences are outside the scope of
the minimum viable demo.

## VAE model

The convolutional VAE implementation is in `src/world_model/vae.py`. It accepts
normalized RGB tensors in NCHW layout with shape `B×3×64×64`, and returns a
reconstruction plus the latent Gaussian mean and log variance. Use
`sample=False` to decode the posterior mean deterministically.

## MDN-RNN model

The recurrent dynamics model is in `src/world_model/mdn_rnn.py`. It consumes
batch-first latent and action sequences, maintains the LSTM hidden and cell
states, and predicts a five-component Gaussian mixture for the next latent
vector. `step()` advances a single frame; its temperature parameter controls
mixture selection and sampling variance, with temperature zero selecting the
most likely component mean.

## Requirements

- Python 3.10 or newer
- Install project dependencies with `pip install -e .`

## Project layout

Package code lives in `src/world_model/`, with runnable scripts in `scripts/`.
Datasets and trained weights are not stored in the repository.

## Collecting trajectories

Run the collector from the repository root:

```powershell
python scripts/collect_data.py --episodes 20 --output data/carracing
```

The collector writes one compressed `.npz` file per episode. Each file contains
`observations` (`uint8`, shape `T×64×64×3`), `actions` (`float32`, shape
`T×3`), `rewards` (`float32`, shape `T`), `dones` (`bool`, shape `T`), and
basic environment metadata. The final `dones` entry marks the episode boundary.
The output directory must not already contain files named `episode_*.npz`.

Use `--render` to display the environment while collecting. This is slower, but
useful for visually checking the exploratory driving behavior.

## Training the VAE

Train on the collected episode files with:

```powershell
python scripts/train_vae.py --data data/carracing --epochs 20
```

The trainer keeps one episode in memory at a time, reserves complete episodes
for validation, and writes the best checkpoint to `artifacts/vae/vae.pt`. It
also writes `reconstructions.png`, with each original frame next to its
deterministic reconstruction, for visual inspection. Use `--device cpu` or
`--device cuda` to choose the compute device explicitly.

## Encoding latent sequences

After training the VAE, encode the dataset using its posterior means:

```powershell
python scripts/encode_latents.py --data data/carracing --checkpoint artifacts/vae/vae.pt --output data/latents
```

This deterministic encoding step writes one file per source episode, preserving
the matching actions and episode-boundary flags for recurrent-model training.

## Training the MDN-RNN

Train the recurrent dynamics model on those latent episodes:

```powershell
python scripts/train_mdn_rnn.py --data data/latents --epochs 30
```

Training uses non-overlapping 64-step windows within each episode, so no
sequence crosses an episode boundary. The best checkpoint is written to
`artifacts/mdn_rnn/mdn_rnn.pt`, and per-epoch negative log likelihood values are
stored in `history.csv`.

## Running the world engine

`src/world_model/engine.py` combines the trained checkpoints for inference. It
uses an encoded episode as a reset seed, then generates frames from actions
without stepping Gymnasium:

```python
from world_model.engine import WorldEngine

engine = WorldEngine()
frame = engine.reset(seed=7)
frame = engine.step([0.0, 0.8, 0.0])  # steering, gas, brake
```

`frame` is an RGB `uint8` array of shape `64×64×3`. Call
`engine.set_temperature(value)` to adjust sampling while the dream is running.
