"""End-to-end check of the HW2 plumbing (data -> train -> evaluate -> report assets).

Uses a tiny synthetic WikiArt zip, a stand-in model, and stand-in steps and losses, so it
runs before your GAN exists. The stand-ins are NOT a GAN (nothing is trained); they only
have the same interface.
"""

import math

import pytest
import torch
from torch import nn

from dgm.data.synthetic import make_fake_wikiart
from dgm.data.wikiart import archive_path, prepare
from dgm.training.loggers import read_csv
from dgm.training.runs import read_json
from dgm.utils.config import load_config
from hw2 import evaluate as hw2_evaluate
from hw2 import report_assets
from hw2 import train as train_module
from hw2.config import Config
from hw2.evaluate import evaluate
from hw2.train import train

SIZE = 16


class StandIn(nn.Module):
    """Linear generator and discriminator with the GAN interface."""

    def __init__(self, image_size: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim, self.image_size = latent_dim, image_size
        pixels = 3 * image_size * image_size
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, pixels), nn.Tanh(), nn.Unflatten(1, (3, image_size, image_size))
        )
        self.discriminator = nn.Sequential(nn.Flatten(), nn.Linear(pixels, 1), nn.Flatten(0))

    @torch.no_grad()
    def sample(self, n, device=None):
        z = torch.randn(n, self.latent_dim, device=device)
        return (self.generator(z) + 1) / 2


def standin_dstep(model, x_real, optimizer, *, loss):
    with torch.no_grad():
        real = torch.sigmoid(model.discriminator(x_real)).mean().item()
    return {"d_loss": 1.0, "d_real": real, "d_fake": 0.5}


def standin_gstep(model, batch_size, optimizer, *, loss):
    return {"g_loss": 1.0}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DGM_RUNS_DIR", str(tmp_path / "runs"))
    make_fake_wikiart(archive_path(), counts={"train": 40, "test": 12})
    prepare(SIZE, val_fraction=0.25)
    monkeypatch.setattr(train_module, "discriminator_step", standin_dstep)
    monkeypatch.setattr(train_module, "generator_step", standin_gstep)
    monkeypatch.setattr(train_module, "discriminator_loss", lambda r, f, kind: (r - f).mean())
    monkeypatch.setattr(train_module, "generator_loss", lambda f, kind: -f.mean())
    monkeypatch.setattr(report_assets, "GENERATED", tmp_path / "generated")
    return tmp_path


def smoke_config(**overrides):
    values = {
        "data.image_size": SIZE,
        "data.batch_size": 4,
        "train.epochs": 2,
        "train.sample_every": 1,
        "train.fid_every": 0,
        "train.device": "cpu",
        "eval.batch_size": 4,
        "eval.num_samples": 8,
        "eval.repeats": 1,
        **overrides,
    }
    return load_config(Config, overrides=[f"{k}={v}" for k, v in values.items()])


def test_train_evaluate_export(workspace):
    run_dir = train(smoke_config(), model=StandIn(SIZE))

    metrics = read_csv(run_dir / "metrics.csv")
    assert list(metrics["epoch"]) == [1.0, 2.0]
    assert {"train/d_loss", "train/g_loss", "val/d_real", "val/fid_2048"} <= set(metrics)
    assert all(math.isnan(v) for v in metrics["val/fid_2048"]), "FID is off in this config"
    for name in ("config.yaml", "env.json", "steps.csv", "summary.json", "loss_curves.pdf"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "samples" / "epoch_002.png").exists()
    assert list((run_dir / "tensorboard").glob("events.out.tfevents.*")), "no TensorBoard log"
    env = read_json(run_dir / "env.json")
    assert env["parameters_generator"] > 0 and env["images"]["train"] > 0
    assert read_json(run_dir / "summary.json")["best_val_fid_2048"] is None

    results = evaluate(run_dir, model=StandIn(SIZE), compute_scores=False)
    for name in ("samples.png", "interpolation.png", "real.png"):
        assert (run_dir / "eval" / name).exists(), name
    assert results["num_samples"] == 8

    report_assets.export(run_dir)
    generated = report_assets.GENERATED
    for name in (
        "loss_curves.pdf",
        "samples.png",
        "interpolation.png",
        "hparams.tex",
        "results.tex",
    ):
        assert (generated / name).exists(), name
    assert r"\newcommand{\ResultTrainImages}" in (generated / "results.tex").read_text()


def test_ema_weights_are_saved_and_used(workspace, monkeypatch):
    run_dir = train(smoke_config(**{"train.ema_decay": 0.9}), model=StandIn(SIZE))
    last = torch.load(run_dir / "checkpoints" / "last.pt", weights_only=True)
    best = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True)
    assert {"ema", "optimizer_g", "optimizer_d"} <= set(last) and "ema" in best
    assert set(last["ema"]) == set(last["model"])

    monkeypatch.setattr(hw2_evaluate, "build_model", lambda cfg: StandIn(SIZE))
    results = evaluate(run_dir, compute_scores=False)
    assert results["weights"] == "ema"
    results = evaluate(run_dir, ["eval.use_ema=false"], compute_scores=False)
    assert results["weights"] == "raw"


@pytest.mark.parametrize(("stop_after", "epochs_run"), [(2, 2), (0, 4)])
def test_dead_game_stops_only_when_asked(workspace, monkeypatch, stop_after, epochs_run):
    # D always wins (D(G(z)) = 0): with stop_dead_epochs=2 the run ends after 2 epochs and still
    # saves the last sample grid; with 0 (the default) it runs all epochs.
    def dead_dstep(model, x_real, optimizer, *, loss):
        return {"d_loss": 0.0, "d_real": 1.0, "d_fake": 0.0}

    monkeypatch.setattr(train_module, "discriminator_step", dead_dstep)
    cfg = smoke_config(**{"train.epochs": 4, "train.stop_dead_epochs": stop_after})
    run_dir = train(cfg, model=StandIn(SIZE))
    summary = read_json(run_dir / "summary.json")
    assert summary["epochs_completed"] == epochs_run
    assert summary["stopped_early"] == (epochs_run < 4)
    assert list(read_csv(run_dir / "metrics.csv")["epoch"]) == [
        float(e) for e in range(1, epochs_run + 1)
    ]
    assert (run_dir / "samples" / f"epoch_{epochs_run:03d}.png").exists()


def test_placeholders(workspace):
    report_assets.write_placeholders()
    text = (report_assets.GENERATED / "results.tex").read_text()
    assert r"\newcommand{\ResultFIDpool}{\textcolor{red}{??}}" in text
    assert (report_assets.GENERATED / "hparams.tex").exists()


@pytest.mark.slow
def test_validation_fid_and_final_scores(workspace):
    cfg = smoke_config(**{"train.fid_every": 1, "train.fid_samples": 6, "eval.repeats": 2})
    run_dir = train(cfg, model=StandIn(SIZE))
    metrics = read_csv(run_dir / "metrics.csv")
    assert all(math.isfinite(v) for v in metrics["val/fid_2048"])
    assert read_json(run_dir / "summary.json")["best_val_fid_2048"] is not None

    results = evaluate(run_dir, model=StandIn(SIZE))
    for score in ("fid", "fid_2048", "inception_score"):
        assert len(results["samples"][score]["values"]) == 2
        assert score in results["real_train_images"]
    evaluate(run_dir, model=StandIn(SIZE), compute_scores=False)  # keeps the saved scores
    assert "samples" in read_json(run_dir / "eval" / "metrics.json")

    report_assets.export(run_dir)
    text = (report_assets.GENERATED / "results.tex").read_text()
    assert "??" not in text, text
