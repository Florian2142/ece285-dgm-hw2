"""Checks for YOUR GAN (models/gan.py), losses (losses.py), and training round (train.py).

Every test is skipped while the code it needs still raises NotImplementedError, so
`uv run pytest homeworks/hw2` shows your progress: skipped = TODO, passed = done.
Check one TODO at a time with -k, for example `uv run pytest homeworks/hw2 -k dloss`.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hw2.config import DataConfig, ModelConfig
from hw2.losses import LOSSES, discriminator_loss, generator_loss
from hw2.models import GAN, Discriminator, Generator
from hw2.train import discriminator_step, generator_step

IMAGE_SIZE = DataConfig().image_size
LATENT = 16
CFG = ModelConfig(latent_dim=LATENT, g_channels=16, d_channels=16)


def todo(fn, *args, **kwargs):
    """Call ``fn``; skip the test (instead of failing) while it is not implemented."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as err:
        pytest.skip(f"TODO: {err}")


@pytest.fixture
def model():
    torch.manual_seed(0)
    return todo(GAN, CFG, image_size=IMAGE_SIZE)


@pytest.fixture
def images():
    x = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(0))
    return 2 * x - 1


def logits(seed, n=6):
    return 3 * torch.randn(n, generator=torch.Generator().manual_seed(seed))


def snapshot(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}


def changed(module, before):
    return [name for name, p in module.named_parameters() if not torch.equal(p, before[name])]


# --- losses.py ----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["minimax", "nonsaturating"])
def test_dloss_is_binary_cross_entropy(kind):
    real, fake = logits(1), logits(2)
    expected = F.binary_cross_entropy_with_logits(
        real, torch.ones_like(real)
    ) + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    loss = todo(discriminator_loss, real, fake, kind)
    assert loss.shape == (), "one number: the batch means"
    torch.testing.assert_close(loss, expected)


def test_dloss_hinge():
    real, fake = logits(3), logits(4)
    expected = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    torch.testing.assert_close(todo(discriminator_loss, real, fake, "hinge"), expected)


def test_gloss_minimax_is_the_g_part_of_v():
    fake = logits(5)
    expected = torch.log(1 - torch.sigmoid(fake)).mean()
    torch.testing.assert_close(todo(generator_loss, fake, "minimax"), expected)


def test_gloss_nonsaturating():
    fake = logits(6)
    expected = -torch.log(torch.sigmoid(fake)).mean()
    torch.testing.assert_close(todo(generator_loss, fake, "nonsaturating"), expected)


def test_gloss_hinge():
    fake = logits(7)
    torch.testing.assert_close(todo(generator_loss, fake, "hinge"), -fake.mean())


@pytest.mark.parametrize("kind", LOSSES)
def test_dloss_and_gloss_are_stable_for_large_logits(kind):
    real = torch.tensor([100.0, -100.0], requires_grad=True)
    fake = torch.tensor([-100.0, 100.0], requires_grad=True)
    d = todo(discriminator_loss, real, fake, kind)
    g = todo(generator_loss, fake, kind)
    assert torch.isfinite(d) and torch.isfinite(g), "use the logits directly, not log(sigmoid(.))"
    (d + g).backward()
    assert torch.isfinite(real.grad).all() and torch.isfinite(fake.grad).all()


def test_dloss_and_gloss_reject_unknown_kinds():
    with pytest.raises(ValueError):
        todo(discriminator_loss, logits(8), logits(9), "wasserstein-ish")
    with pytest.raises(ValueError):
        todo(generator_loss, logits(8), "wasserstein-ish")


def test_gloss_nonsaturating_keeps_the_gradient_when_d_wins():
    """D(G(z)) = sigmoid(-8) ~ 0.0003: the minimax gradient vanishes, the nonsaturating one not."""
    grads = {}
    for kind in ("minimax", "nonsaturating"):
        fake = torch.full((1,), -8.0, requires_grad=True)
        todo(generator_loss, fake, kind).backward()
        grads[kind] = fake.grad.abs().item()
    assert grads["minimax"] < 1e-3 < 0.9 < grads["nonsaturating"]


# --- models/gan.py --------------------------------------------------------------------


def test_generator_shapes_and_range():
    torch.manual_seed(0)
    generator = todo(Generator, CFG, IMAGE_SIZE)
    x = todo(generator, torch.randn(4, LATENT))
    assert x.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert x.min() >= -1.0 and x.max() <= 1.0, "the generator's images live in [-1, 1]"


def test_discriminator_shapes(images):
    torch.manual_seed(0)
    discriminator = todo(Discriminator, CFG, IMAGE_SIZE)
    out = todo(discriminator, images)
    assert out.shape == (len(images),), "one logit per image, shape [B]"


def test_discriminator_updates_batchnorm_once_per_forward(images):
    discriminator = todo(Discriminator, CFG, IMAGE_SIZE).train()
    norms = [m for m in discriminator.modules() if isinstance(m, nn.BatchNorm2d)]
    before = [m.num_batches_tracked.item() for m in norms]
    todo(discriminator, images)
    assert norms, "the baseline discriminator uses BatchNorm"
    assert [m.num_batches_tracked.item() - n for m, n in zip(norms, before, strict=True)] == [1] * len(norms)


def test_init_weights_dcgan(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d | nn.ConvTranspose2d | nn.Linear):
            weight = module.weight.detach()
            if weight.numel() >= 1000:
                assert 0.017 < weight.std() < 0.023, f"{name}.weight: std should be 0.02"
                assert weight.mean().abs() < 0.005, f"{name}.weight: mean should be 0"
            if module.bias is not None:
                assert torch.all(module.bias == 0), f"{name}.bias should be 0"
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine:
            assert (module.weight - 1).abs().max() < 0.12, f"{name}.weight: ~ N(1, 0.02^2)"
            assert torch.all(module.bias == 0), f"{name}.bias should be 0"


def test_sample_draws_new_images_in_unit_range(model):
    model.eval()
    samples = todo(model.sample, 6)
    assert samples.shape == (6, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert samples.min() >= 0.0 and samples.max() <= 1.0, "sample returns images in [0, 1]"
    assert not torch.allclose(samples[0], samples[1]), "different z should give different images"


def test_gradients_reach_every_parameter(model, images):
    fakes = todo(model.generator, torch.randn(len(images), LATENT))
    d_loss = todo(
        discriminator_loss, model.discriminator(images), model.discriminator(fakes.detach())
    )
    d_loss.backward()
    g_loss = todo(generator_loss, model.discriminator(fakes))
    g_loss.backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


# --- train.py: one round of the game -----------------------------------------------------


def optimizers(model):
    return (
        torch.optim.Adam(model.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999)),
        torch.optim.Adam(model.generator.parameters(), lr=1e-3, betas=(0.5, 0.999)),
    )


def test_dstep_updates_only_the_discriminator(model, images):
    model.train()
    optimizer_d, _ = optimizers(model)
    g_before, d_before = snapshot(model.generator), snapshot(model.discriminator)
    terms = todo(discriminator_step, model, images, optimizer_d, loss="nonsaturating")
    assert set(terms) == {"d_loss", "d_real", "d_fake"}
    assert all(isinstance(v, float) for v in terms.values()), "return plain floats (.item())"
    assert 0 <= terms["d_real"] <= 1 and 0 <= terms["d_fake"] <= 1, "D outputs are probabilities"
    assert changed(model.discriminator, d_before), "D was not updated"
    assert not changed(model.generator, g_before), "G must not change in the D step"
    with_grad = [
        n for n, p in model.generator.named_parameters() if p.grad is not None and p.grad.any()
    ]
    assert not with_grad, f"the D step filled G's gradients ({with_grad[:3]}...): detach the fakes"


def test_gstep_updates_only_the_generator(model):
    model.train()
    _, optimizer_g = optimizers(model)
    g_before, d_before = snapshot(model.generator), snapshot(model.discriminator)
    terms = todo(generator_step, model, 4, optimizer_g, loss="nonsaturating")
    assert set(terms) == {"g_loss"} and isinstance(terms["g_loss"], float)
    assert changed(model.generator, g_before), "G was not updated"
    assert not changed(model.discriminator, d_before), "D must not change in the G step"


class ToyGAN(nn.Module):
    """1-D data: small MLPs with the GAN interface (generator, discriminator, latent_dim)."""

    def __init__(self, latent_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.discriminator = nn.Sequential(
            nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1), nn.Flatten(0)
        )


@pytest.mark.slow
@pytest.mark.parametrize("kind", LOSSES)
def test_game_learns_a_1d_gaussian(kind):
    """Your steps and losses together: G must move N(0, 1)-ish noise to the data N(4, 0.5^2)."""
    torch.manual_seed(0)
    toy = ToyGAN()
    optimizer_d = torch.optim.Adam(toy.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(toy.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    for _ in range(2000):
        x_real = 4 + 0.5 * torch.randn(128, 1)
        todo(discriminator_step, toy, x_real, optimizer_d, loss=kind)
        todo(generator_step, toy, 128, optimizer_g, loss=kind)
    with torch.no_grad():
        fake = toy.generator(torch.randn(4000, toy.latent_dim))
    assert abs(fake.mean().item() - 4) < 0.4, f"mean {fake.mean():.2f}, want 4"
    assert 0.15 < fake.std().item() < 1.5, f"std {fake.std():.2f}, want 0.5"
