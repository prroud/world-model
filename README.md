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

## Requirements

- Python 3.10 or newer
- Install project dependencies with `pip install -e .`

## Project layout

Package code lives in `src/world_model/`, with runnable scripts in `scripts/`.
Datasets and trained weights are not stored in the repository.
