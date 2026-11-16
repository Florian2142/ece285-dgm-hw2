"""Evaluate a trained HW2 run: new paintings, a latent interpolation, FID and Inception Score.

    uv run python -m hw2.evaluate                       # latest run, best checkpoint
    uv run python -m hw2.evaluate --run 20261110-101500-gan-base eval.repeats=5
    uv run python -m hw2.evaluate --no-scores           # image grids only (fast; keeps earlier scores)

Writes <run>/eval/:
    samples.png          64 new paintings, z ~ N(0, I)
    interpolation.png    8 rows: G(z) along a straight line from one random z to another
    real.png             64 reference paintings, for comparison
    metrics.json         all scores (mean, std, every repeat) and the settings used

Scores (details in dgm/evaluation/metrics.py): ``fid`` is ignite's default FID exactly as in the
course tutorial (it compares Inception class probabilities, so values are small); ``fid_2048``
is the standard FID on 2048-d pool features; ``inception_score`` is ignite's default IS.

Protocol (describe it in the report):
- samples: n = eval.num_samples generated paintings vs. n random reference paintings
  (eval.reference_split), both drawn fresh in each of eval.repeats repeats -> mean and std
- real baseline: n random TRAINING paintings vs. n reference paintings: what a perfect
  generator would score at this n (FID has a positive bias for finite n)
- weights: the EMA weights when the run has them and eval.use_ema is true
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from dgm.data.wikiart import WikiArt
from dgm.evaluation.metrics import InceptionScorer
from dgm.training.checkpoint import load_checkpoint
from dgm.training.runs import resolve_run_dir, write_json
from dgm.utils.config import load_config
from dgm.utils.misc import get_device, seed_everything
from dgm.utils.viz import save_image_grid
from hw2.config import Config
from hw2.models import to_image_range
from hw2.train import EXPERIMENT, build_model, generate, random_subset

SCORES = {"fid": "FID (tutorial)", "fid_2048": "FID-2048", "inception_score": "IS"}


@torch.no_grad()
def interpolate(model: nn.Module, rows: int, steps: int, seed: int, device: torch.device) -> Tensor:
    """``rows`` lines z0 -> z1 in latent space, ``steps`` images each: ``[rows * steps, 3, H, W]``."""
    gen = torch.Generator().manual_seed(seed)
    z0, z1 = (torch.randn(rows, 1, model.latent_dim, generator=gen) for _ in range(2))
    t = torch.linspace(0, 1, steps)[None, :, None]
    z = ((1 - t) * z0 + t * z1).reshape(rows * steps, model.latent_dim)
    return to_image_range(model.generator(z.to(device))).cpu()


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def load_run(run_dir: Path, cfg: Config, device: torch.device) -> tuple[nn.Module, str]:
    """The run's GAN with the chosen checkpoint; EMA weights if the run has them."""
    model = build_model(cfg)
    state = load_checkpoint(
        run_dir / "checkpoints" / f"{cfg.eval.checkpoint}.pt", model, map_location=device
    )
    weights = "raw"
    if cfg.eval.use_ema and "ema" in state:
        model.load_state_dict(state["ema"])
        weights = "ema"
    return model, weights


def evaluate(
    run_dir: Path,
    overrides: list[str] | None = None,
    *,
    model: nn.Module | None = None,
    compute_scores: bool = True,
) -> dict:
    cfg = load_config(Config, run_dir / "config.yaml", overrides or [])
    device = get_device(cfg.train.device)
    weights = "given"
    if model is None:
        model, weights = load_run(run_dir, cfg, device)
    model.to(device).eval()
    seed_everything(cfg.eval.seed)
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    reference = WikiArt(cfg.eval.reference_split, cfg.data.image_size, cfg.data.styles).as_float()
    n = cfg.eval.num_samples
    if n > len(reference):
        raise ValueError(
            f"eval.num_samples={n} exceeds the {len(reference)} '{cfg.eval.reference_split}' paintings"
        )

    save_image_grid(model.sample(64, device), out_dir / "samples.png", nrow=8)
    save_image_grid(
        interpolate(model, 8, 10, cfg.eval.seed, device), out_dir / "interpolation.png", nrow=10
    )
    save_image_grid(random_subset(reference, 64, cfg.eval.seed), out_dir / "real.png", nrow=8)

    results: dict = {
        "run": run_dir.name,
        "checkpoint": cfg.eval.checkpoint,
        "weights": weights,
        "reference_split": cfg.eval.reference_split,
        "num_samples": n,
        "repeats": cfg.eval.repeats,
        "seed": cfg.eval.seed,
        "preprocessing": "PIL bilinear 299x299, RGB, [0,1]; torchvision Inception-v3",
        "scores": {
            "fid": "ignite FID() default as in the tutorial: 1000-d class probabilities",
            "fid_2048": "ignite FID on 2048-d pool features (standard definition)",
            "inception_score": "ignite InceptionScore() default",
        },
    }
    if compute_scores:
        scorer = InceptionScorer(device, batch_size=cfg.eval.batch_size)
        per_repeat: dict[str, list[float]] = {name: [] for name in SCORES}
        for repeat in range(cfg.eval.repeats):
            real = random_subset(reference, n, cfg.eval.seed + repeat)
            scores = scorer(generate(model, n, cfg.eval.batch_size, device), real)
            for name in SCORES:
                per_repeat[name].append(scores[name])
            shown = "  ".join(f"{SCORES[k]} {fmt(v)}" for k, v in scores.items())
            print(f"repeat {repeat + 1}/{cfg.eval.repeats}: {shown}", flush=True)
        results["samples"] = {name: summarize(values) for name, values in per_repeat.items()}
        if cfg.eval.real_baseline and cfg.eval.reference_split != "train":
            training = WikiArt("train", cfg.data.image_size, cfg.data.styles).as_float()
            if len(training) >= n:
                real = random_subset(reference, n, cfg.eval.seed)
                results["real_train_images"] = scorer(
                    random_subset(training, n, cfg.eval.seed), real
                )
        write_json(out_dir / "metrics.json", results)  # only scored runs replace earlier scores
    print_summary(results)
    return results


def fmt(value: float, like: float | None = None) -> str:
    """Enough digits for both scales: tutorial FID (< 1) and FID-2048 (tens to hundreds).

    ``like``: use the decimals that suit this value instead (e.g. the mean, for a std).
    """
    ref = abs(value if like is None else like)
    return f"{value:.{4 if ref < 1 else 2 if ref < 100 else 1}f}"


def print_summary(results: dict) -> None:
    if "samples" not in results:
        print(f"saved image grids for {results['run']} (scores skipped)")
        return
    print(
        f"\n{results['run']} ({results['weights']} weights): {results['num_samples']} images vs. "
        f"'{results['reference_split']}', {results['repeats']} repeat(s)"
    )
    print(f"  {'':<17}" + "".join(f"{label:>22}" for label in SCORES.values()))
    samples = results["samples"]
    cells = [
        f"{fmt(samples[k]['mean'])} +- {fmt(samples[k]['std'], like=samples[k]['mean'])}"
        for k in SCORES
    ]
    print(f"  {'samples':<17}" + "".join(f"{cell:>22}" for cell in cells))
    if "real_train_images" in results:
        real = results["real_train_images"]
        print(f"  {'real (train)':<17}" + "".join(f"{fmt(real[k]):>22}" for k in SCORES))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", default="latest", help="run name or folder under runs/hw2 (default: latest)"
    )
    parser.add_argument("--no-scores", action="store_true", help="only save the image grids")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="override eval settings, e.g. eval.repeats=5",
    )
    args = parser.parse_args(argv)
    try:
        run_dir = resolve_run_dir(EXPERIMENT, args.run)
        evaluate(run_dir, args.overrides, compute_scores=not args.no_scores)
    except FileNotFoundError as err:
        raise SystemExit(f"error: {err}") from None
    except NotImplementedError as err:
        raise SystemExit(
            f"\nNot implemented yet: {err}\nFinish homeworks/hw2/models/gan.py first."
        ) from None


if __name__ == "__main__":
    main()
