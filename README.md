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
