import torch
import pytest

from world_model.mdn_rnn import MDNRNN, mixture_nll
from world_model.vae import VAE


def test_vae_reconstructs_batch_and_deterministic_mean() -> None:
    model = VAE(latent_dim=4).eval()
    images = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        reconstruction, mean, logvar = model(images, sample=False)

    assert reconstruction.shape == images.shape
    assert mean.shape == (1, 4)
    assert logvar.shape == (1, 4)
    assert torch.all((reconstruction >= 0) & (reconstruction <= 1))


def test_vae_rejects_wrong_image_shape() -> None:
    model = VAE(latent_dim=4)

    with pytest.raises(ValueError, match="64, 64"):
        model.encode(torch.zeros(1, 3, 32, 32))


def test_mdn_rnn_shapes_loss_and_deterministic_step() -> None:
    model = MDNRNN(latent_dim=4, hidden_size=8, num_mixtures=3).eval()
    latents = torch.randn(2, 5, 4)
    actions = torch.rand(2, 5, 3)

    with torch.inference_mode():
        output = model(latents, actions)
        loss = mixture_nll(output, torch.randn(2, 5, 4))
        first, state = model.step(latents[:, 0], actions[:, 0], temperature=0)
        second, _ = model.step(latents[:, 0], actions[:, 0], temperature=0)

    assert output.mixture_logits.shape == (2, 5, 3)
    assert output.means.shape == (2, 5, 3, 4)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert first.shape == (2, 4)
    assert torch.equal(first, second)
    assert state[0].shape == (1, 2, 8)


def test_mdn_rnn_rejects_misaligned_sequences() -> None:
    model = MDNRNN(latent_dim=4, hidden_size=8)

    with pytest.raises(ValueError, match="matching batch and time"):
        model(torch.zeros(1, 3, 4), torch.zeros(1, 2, 3))
