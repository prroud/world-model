import pytest
import torch

from world_model.mdn_rnn import MDNRNN, mixture_nll
from world_model.vae import VAE


def test_vae_reconstructs_batch_with_expected_shape_and_range() -> None:
    model = VAE(latent_dim=8).eval()
    images = torch.rand(2, 3, 64, 64)

    with torch.inference_mode():
        reconstruction, mean, log_variance = model(images, sample=False)

    assert reconstruction.shape == images.shape
    assert mean.shape == (2, 8)
    assert log_variance.shape == (2, 8)
    assert torch.all((reconstruction >= 0) & (reconstruction <= 1))


def test_vae_rejects_wrong_image_shape() -> None:
    model = VAE(latent_dim=8)

    with pytest.raises(ValueError, match="64, 64"):
        model.encode(torch.rand(2, 3, 32, 32))


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
    assert output.log_stds.shape == (2, 5, 3, 4)
    assert torch.isfinite(loss)
    assert first.shape == (2, 4)
    assert torch.equal(first, second)
    assert state[0].shape == (1, 2, 8)


def test_mixture_nll_is_differentiable() -> None:
    model = MDNRNN(latent_dim=4, hidden_size=8, num_mixtures=2)
    output = model(torch.randn(2, 3, 4), torch.randn(2, 3, 3))
    loss = mixture_nll(output, torch.randn(2, 3, 4))

    loss.backward()

    assert torch.isfinite(loss)
    assert model.mean_head.weight.grad is not None
    assert torch.isfinite(model.mean_head.weight.grad).all()


def test_mdn_rnn_rejects_misaligned_sequences() -> None:
    model = MDNRNN(latent_dim=4, hidden_size=8)

    with pytest.raises(ValueError, match="matching batch and time"):
        model(torch.zeros(1, 3, 4), torch.zeros(1, 2, 3))


def test_mixture_nll_rejects_mismatched_targets() -> None:
    model = MDNRNN(latent_dim=4, hidden_size=8, num_mixtures=2)
    output = model(torch.randn(2, 3, 4), torch.randn(2, 3, 3))

    with pytest.raises(ValueError, match="targets must match"):
        mixture_nll(output, torch.randn(2, 3, 5))
