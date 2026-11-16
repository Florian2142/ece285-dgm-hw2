"""Train the HW2 GAN on WikiArt paintings.   >>> The two step functions are your homework. <<<

    uv run python -m hw2.train                                   # configs/base.yaml
    uv run python -m hw2.train --config homeworks/hw2/configs/smoke.yaml
    uv run python -m hw2.train train.epochs=50 train.loss=hinge name=gan-hinge

One round of the game per training batch, as in Algorithm 1 of Goodfellow et al. (2014):
update D on real and fake images (``discriminator_step``), then update G
(``generator_step``). After every epoch: the same losses on the validation split without
updates (``validation_step``); every ``train.fid_every`` epochs a validation FID, which picks
``best.pt``. Watch a run live: ``uv run tensorboard --logdir runs/hw2``.

Check your two steps in this order:

    1. discriminator_step    uv run pytest homeworks/hw2 -k dstep
    2. generator_step        uv run pytest homeworks/hw2 -k gstep

Each run writes runs/hw2/<timestamp>-<name>/:
    config.yaml        the exact settings (re-load it to reproduce the run)
    env.json           versions, GPU, git revision, parameter and image counts
    metrics.csv        one row per epoch: train/val losses, D outputs, validation FID, time
    steps.csv          one row per training batch
    tensorboard/       the same numbers plus sample images, for TensorBoard
    checkpoints/       last.pt and best.pt (lowest validation FID-2048; the last epoch if FID is off)
    samples/           epoch_XXX.png: the same 64 fixed z after each epoch
    loss_curves.pdf    the curves, updated every epoch (also .png)
    summary.json       best epoch, final numbers, training time
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dgm.data.wikiart import WikiArt
from dgm.evaluation.metrics import InceptionScorer
from dgm.training.checkpoint import save_checkpoint
from dgm.training.loggers import CSVLogger, create_tqdm_bar, read_csv
from dgm.training.runs import create_run_dir, write_json
from dgm.utils.config import add_config_args, load_config, save_config
from dgm.utils.misc import count_parameters, environment_info, get_device, seed_everything
from dgm.utils.viz import plot_metric_panels, save_image_grid
from hw2.config import Config, TrainConfig
from hw2.losses import discriminator_loss, generator_loss
from hw2.models import GAN, to_image_range, to_model_range

EXPERIMENT = "hw2"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "base.yaml"
# Training curves = means over each epoch. The validation losses are one snapshot at the end of an
# epoch (single points can jump to 4.5 and squash the training curves); they stay in metrics.csv,
# TensorBoard, and the notebook's plot_history.
CURVES = (
    ("Discriminator loss", (("train/d_loss", "train (epoch mean)"),)),
    ("Generator loss", (("train/g_loss", "train (epoch mean)"),)),
    ("D output (train)", (("train/d_real", "D(x)"), ("train/d_fake", "D(G(z))"))),
    ("FID-2048 (val)", (("val/fid_2048", "val"),)),
)
NO_FID = {"fid": math.nan, "fid_2048": math.nan, "inception_score": math.nan}
DEAD_D_FAKE = 1e-6  # mean D(G(z)) of an epoch below this = a dead game (train.stop_dead_epochs)

Batch = tuple[Tensor, Tensor]


# ---------------------------------------------------------------------------------------
# One round of the game: YOUR CODE. Everything below this section runs it.
# ---------------------------------------------------------------------------------------


def discriminator_step(
    model: GAN, x_real: Tensor, optimizer: torch.optim.Optimizer, *, loss: str
) -> dict[str, float]:
    """One update of D on real images ``x_real [B, 3, H, W]`` in [-1, 1] and B fresh fakes.

    Returns plain floats: ``"d_loss"``, and ``"d_real"`` / ``"d_fake"``, the mean probability
    that D gives the real and the fake images (sigmoid of the logits, before the update).
    """

    ########################################################################
    # TODO: Draw B fakes from G, score real and fake with D, and take one  #
    # optimizer step on discriminator_loss (losses.py). G must not change. #
    #                                                                      #
    # Question for you: without fake.detach(), backward() also fills the   #
    # .grad of G's weights. What would that do to the next generator       #
    # step, and what does detach() save?                                   #
    ########################################################################

    

    z = torch.randn(x_real.size(0), model.latent_dim, device=x_real.device) # sample
    # Note we detach for speed not for correcntess since we have two optimizers
    fake = model.generator(z).detach() # cut the graph: new tensor, same numbers, grad_fn = None, so backward never reaches G

    # NOTE: Must be the same output shape
    fake_logits = model.discriminator(fake)
    true_logits = model.discriminator(x_real)

    i_loss = discriminator_loss(real_logits=true_logits, fake_logits=fake_logits, kind=loss)

    # the training
    optimizer.zero_grad()
    i_loss.backward()
    optimizer.step()

    return {
    "d_loss": i_loss.item(),
    "d_real": torch.sigmoid(true_logits).mean().item(),
    "d_fake": torch.sigmoid(fake_logits).mean().item(),
    }



    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


def generator_step(
    model: GAN, batch_size: int, optimizer: torch.optim.Optimizer, *, loss: str
) -> dict[str, float]:
    """One update of G with ``batch_size`` fresh fakes. Returns ``{"g_loss": float}``.

    The noise goes on the model's device: ``next(model.parameters()).device``.
    """

    ########################################################################
    # TODO: Draw fresh fakes, score them with D, and take one optimizer    #
    # step on generator_loss (losses.py). D must not change.               #
    #                                                                      #
    # Question for you: the gradient of the G loss flows THROUGH D. Why    #
    # does D still not change, and why must the next discriminator step    #
    # start with zero_grad?                                                #
    ########################################################################

    device = next(model.parameters()).device

    # same as above
    z = torch.randn(batch_size, model.latent_dim, device=device) # sample
    fake = model.generator(z) # keep it attached

    fake_logits = model.discriminator(fake) # score the fake ones

    g_loss = generator_loss(fake_logits=fake_logits, kind=loss)

    optimizer.zero_grad()
    g_loss.backward()
    optimizer.step()

    return {
        "g_loss": g_loss.item(),
        }




    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


# ---------------------------------------------------------------------------------------
# Plumbing: data, the loop around your steps, logging, checkpoints.
# ---------------------------------------------------------------------------------------


def build_model(cfg: Config) -> GAN:
    return GAN(cfg.model, image_size=cfg.data.image_size)


def make_loader(split: str, cfg: Config) -> DataLoader:
    """Shuffled batches for every split: the files are sorted by style, and with BatchNorm a
    batch of one style has other statistics than the mixed batches of training."""
    return DataLoader(
        WikiArt(split, cfg.data.image_size, cfg.data.styles),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.train.seed),
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )


def random_hflip(x: Tensor) -> Tensor:
    """Mirror each image with probability 1/2."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


def training_step(
    model: GAN,
    batch: Batch,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    *,
    cfg: TrainConfig,
    device: torch.device,
    ema: AveragedModel | None = None,
) -> dict[str, float]:
    """One round of the game on one batch: ``discriminator_step``, then ``generator_step``."""
    model.train()
    x, _ = batch  # the style labels are not used: the GAN models p(x)
    x = to_model_range(x.to(device, non_blocking=True))
    if cfg.hflip:
        x = random_hflip(x)
    terms = discriminator_step(model, x, optimizer_d, loss=cfg.loss)
    terms |= generator_step(model, len(x), optimizer_g, loss=cfg.loss)
    if ema is not None:
        ema.update_parameters(model)
    return terms


@contextmanager
def frozen_batchnorm_stats(model: nn.Module) -> Iterator[None]:
    """Forward passes in training mode (batch statistics) that keep the running averages."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in norms]
    for m in norms:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in zip(norms, saved, strict=True):
            m.momentum = momentum


@torch.no_grad()
def validation_step(
    model: GAN, batch: Batch, *, loss: str, device: torch.device
) -> dict[str, float]:
    """The losses of one round on held-out paintings and fresh fakes, without any update.

    The networks run as in training (BatchNorm uses batch statistics), so the numbers compare
    with the training ones; weights and running averages stay as they are.
    """
    model.train()
    x = to_model_range(batch[0].to(device, non_blocking=True))
    with frozen_batchnorm_stats(model):
        fake = model.generator(torch.randn(len(x), model.latent_dim, device=device))
        real_logits, fake_logits = model.discriminator(x), model.discriminator(fake)
    return {
        "d_loss": discriminator_loss(real_logits, fake_logits, loss).item(),
        "g_loss": generator_loss(fake_logits, loss).item(),
        "d_real": torch.sigmoid(real_logits).mean().item(),
        "d_fake": torch.sigmoid(fake_logits).mean().item(),
    }


def run_epoch(
    step: Callable[[Batch], dict[str, float]],
    loader: DataLoader,
    *,
    desc: str,
    max_batches: int = 0,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Apply ``step`` to every batch of ``loader``; return the per-image averages of its outputs."""
    total = min(max_batches, len(loader)) if max_batches else len(loader)
    if total == 0:
        raise ValueError(f"{desc}: no batches (is the batch size larger than the split?)")
    totals: dict[str, float] = defaultdict(float)
    seen = 0
    loop = create_tqdm_bar(loader, desc=desc, total=total)
    for i, batch in loop:
        if i >= total:
            break
        values = step(batch)
        bad = {name: value for name, value in values.items() if not math.isfinite(value)}
        if bad:
            raise FloatingPointError(f"{desc}: {bad} in batch {i}: {values}")
        seen += len(batch[0])
        for name, value in values.items():
            totals[name] += value * len(batch[0])
        loop.set_postfix(
            d=f"{totals.get('d_loss', 0) / seen:.3f}", g=f"{totals.get('g_loss', 0) / seen:.3f}"
        )
        if on_step is not None:
            on_step(i, values)
    loop.close()
    return {name: total_value / seen for name, total_value in totals.items()}


@torch.no_grad()
def generate(model: nn.Module, n: int, batch_size: int, device: torch.device) -> Tensor:
    """``n`` new images ``[n, 3, H, W]`` in [0, 1] on the CPU, from ``model.sample`` in eval mode."""
    was_training = model.training
    model.eval()
    chunks = [
        model.sample(min(batch_size, n - start), device).float().cpu()
        for start in range(0, n, batch_size)
    ]
    model.train(was_training)
    return torch.cat(chunks)


@torch.no_grad()
def save_progress_images(sampler: GAN, fixed_z: Tensor, folder: Path, epoch: int) -> Tensor:
    """G(fixed z) in eval mode: the same 64 latents every epoch, so the grids show the progress."""
    was_training = sampler.training
    sampler.eval()
    images = to_image_range(sampler.generator(fixed_z))
    save_image_grid(images, folder / f"epoch_{epoch:03d}.png", nrow=8)
    sampler.train(was_training)
    return images


def random_subset(images: Tensor, n: int, seed: int) -> Tensor:
    """``n`` images in random order (the splits are sorted by style)."""
    return images[torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))[:n]]


def train_model(
    model: GAN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    tb_logger: SummaryWriter,
    *,
    cfg: TrainConfig,
    run_dir: Path,
    device: torch.device,
) -> dict:
    """The training loop: each epoch is one pass over the training set, then the validation set."""
    epoch_log = CSVLogger(run_dir / "metrics.csv")
    step_log = CSVLogger(run_dir / "steps.csv")
    fixed_z = torch.randn(64, model.latent_dim, generator=torch.Generator().manual_seed(cfg.seed))
    fixed_z = fixed_z.to(device)
    ema = None
    if cfg.ema_decay > 0:  # a slowly moving average of the weights; often better samples
        ema = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay), use_buffers=True
        )
    sampler = ema.module if ema is not None else model
    scorer, fid_real = None, None
    if cfg.fid_every > 0:
        fid_real = random_subset(val_loader.dataset.as_float(), cfg.fid_samples, cfg.seed)
        scorer = InceptionScorer(device)

    global_step = 0

    def log_step(batch_index: int, values: dict[str, float]) -> None:
        nonlocal global_step
        step_log.log({"epoch": epoch, "batch": batch_index, **values})
        for name, value in values.items():
            tb_logger.add_scalar(f"step/{name}", value, global_step)
        global_step += 1

    best = {"epoch": 0, "val_fid_2048": math.inf}
    dead_epochs, stopped_early = 0, False
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        # Training: one round of the game on every batch.
        train_stats = run_epoch(
            partial(
                training_step,
                model,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                cfg=cfg,
                device=device,
                ema=ema,
            ),
            train_loader,
            desc=f"Training Epoch [{epoch}/{cfg.epochs}]",
            max_batches=cfg.max_batches_per_epoch,
            on_step=log_step,
        )
        # Validation: the same losses on held-out paintings, no updates.
        val_stats = run_epoch(
            partial(validation_step, model, loss=cfg.loss, device=device),
            val_loader,
            desc=f"Validation Epoch [{epoch}/{cfg.epochs}]",
            max_batches=cfg.max_batches_per_epoch,
        )
        # A dead game: D(G(z)) ~ 0 for the whole epoch, so G gets (almost) no gradient.
        dead_epochs = dead_epochs + 1 if train_stats["d_fake"] < DEAD_D_FAKE else 0
        stopped_early = 0 < cfg.stop_dead_epochs <= dead_epochs and epoch < cfg.epochs
        last_epoch = epoch == cfg.epochs or stopped_early
        scores = NO_FID
        if scorer is not None and (epoch % cfg.fid_every == 0 or last_epoch):
            fake = generate(sampler, len(fid_real), train_loader.batch_size, device)
            scores = scorer(fake, fid_real)
        val_stats |= scores

        elapsed = time.perf_counter() - start
        epoch_log.log(
            {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_stats.items()},
                **{f"val/{k}": v for k, v in val_stats.items()},
                "time_s": round(elapsed, 2),
            }
        )
        # add_scalar, not add_scalars: the latter makes one subfolder per curve, and the
        # long event-file paths can break on Windows.
        for split, stats in (("train", train_stats), ("val", val_stats)):
            for name, value in stats.items():
                if math.isfinite(value):
                    tb_logger.add_scalar(f"{split}/{name}", value, epoch)

        # Checkpoints: always the last epoch; the best one by validation FID-2048.
        extra = {"optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict()}
        if ema is not None:
            extra["ema"] = ema.module.state_dict()
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model=model, epoch=epoch, **extra)
        fid_2048 = scores["fid_2048"]
        if scorer is None or (math.isfinite(fid_2048) and fid_2048 < best["val_fid_2048"]):
            best = {"epoch": epoch, "val_fid_2048": fid_2048}
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model=model,
                epoch=epoch,
                val_fid_2048=fid_2048,
                **({"ema": extra["ema"]} if ema is not None else {}),
            )
        if epoch % cfg.sample_every == 0 or epoch == 1 or last_epoch:
            images = save_progress_images(sampler, fixed_z, run_dir / "samples", epoch)
            tb_logger.add_images("samples/fixed_z", images, epoch)
        plot_metric_panels(read_csv(run_dir / "metrics.csv"), run_dir / "loss_curves", CURVES)

        fid_text = f"  val FID-2048 {fid_2048:.1f}" if math.isfinite(fid_2048) else ""
        print(
            f"epoch {epoch:3d}/{cfg.epochs}  D {train_stats['d_loss']:.3f}  "
            f"G {train_stats['g_loss']:.3f}  D(x) {train_stats['d_real']:.2f}  "
            f"D(G(z)) {train_stats['d_fake']:.2f}{fid_text}  {elapsed:.0f}s",
            flush=True,
        )
        if stopped_early:
            print(
                f"stopped: mean D(G(z)) < {DEAD_D_FAKE:g} for {dead_epochs} epochs in a row "
                f"(a dead game; train.stop_dead_epochs={cfg.stop_dead_epochs})",
                flush=True,
            )
            break

    return {
        "best_epoch": best["epoch"],
        "best_val_fid_2048": best["val_fid_2048"] if scorer is not None else None,
        "final_train": train_stats,
        "final_val": val_stats,
        "epochs_completed": epoch,
        "stopped_early": stopped_early,
        "train_time_s": round(time.perf_counter() - start, 1),
    }


def train(cfg: Config, *, model: nn.Module | None = None, run_dir: Path | None = None) -> Path:
    """Build everything from ``cfg`` (default: your GAN), then train."""
    seed_everything(cfg.train.seed)
    device = get_device(cfg.train.device)
    torch.backends.cudnn.benchmark = True

    train_loader = make_loader("train", cfg)
    val_loader = make_loader("val", cfg)
    model = (model if model is not None else build_model(cfg)).to(device)
    betas = (cfg.train.beta1, cfg.train.beta2)
    # Two players, two optimizers: each one updates only its own network.
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=cfg.train.lr_g, betas=betas)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.train.lr_d, betas=betas)

    run_dir = run_dir if run_dir is not None else create_run_dir(EXPERIMENT, cfg.name)
    save_config(cfg, run_dir / "config.yaml")
    params_g, params_d = count_parameters(model.generator), count_parameters(model.discriminator)
    write_json(
        run_dir / "env.json",
        {
            **environment_info(),
            "device": str(device),
            "parameters": count_parameters(model),
            "parameters_generator": params_g,
            "parameters_discriminator": params_d,
            "images": {"train": len(train_loader.dataset), "val": len(val_loader.dataset)},
        },
    )
    print(f"run folder: {run_dir}")
    print(
        f"device {device} | G {params_g:,} / D {params_d:,} parameters | "
        f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val paintings"
    )

    tb_logger = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    try:
        summary = train_model(
            model,
            train_loader,
            val_loader,
            optimizer_g,
            optimizer_d,
            tb_logger,
            cfg=cfg.train,
            run_dir=run_dir,
            device=device,
        )
    finally:
        tb_logger.close()
    write_json(run_dir / "summary.json", summary)
    best = summary["best_val_fid_2048"]
    best_text = (
        "done" if best is None else f"best val FID-2048 {best:.1f} at epoch {summary['best_epoch']}"
    )
    print(f"{best_text} -> {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(Config, args.config, args.overrides)
    try:
        train(cfg)
    except NotImplementedError as err:
        raise SystemExit(
            f"\nNot implemented yet: {err}\n"
            "Finish the TODOs in homeworks/hw2/models/gan.py, losses.py, and train.py, then check:\n"
            "    uv run pytest homeworks/hw2"
        ) from None


if __name__ == "__main__":
    main()
