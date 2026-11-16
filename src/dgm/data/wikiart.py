"""WikiArt paintings from Kaggle, prepared for generative modeling.

AI-assisted supporting code for downloading, splitting, and preprocessing the data.

Source: https://www.kaggle.com/datasets/steubk/wikiart (WikiArt.org paintings). The zip
(33.7 GB) holds 27 style folders of JPEGs (about 81k files) and ``classes.csv``, which
lists the 80,042 unique paintings. Paintings with 2-3 genres exist as extra copies in
other style folders, so the rows of ``classes.csv`` are the image list, not a glob.

Download once (resumes after network errors; no Kaggle account needed) and cache resized
RGB arrays. The zip is kept, because it is the only raw copy. Images are read straight
from the zip; nothing is extracted:

    uv run python -m dgm.data.wikiart --sizes 64

Splits used in this repo
- test: official subset "test", minus rows whose perceptual hash (phash) also occurs in
  an official train row (near-duplicates would leak into the FID reference set).
- train / val: official subset "train", split by PHASH GROUP (equal phash = same split);
  ``val_fraction`` of the groups go to val; the split is seeded and recorded in meta.json.
- Rows of the subset "uncertain artist" are dropped.
- The label of an image is the index of its style folder in ``meta.json["styles"]``.

Preprocessing: RGB, center crop to a square, resize to ``size x size`` (Lanczos).
Saved as ``data/processed/wikiart/<size>px/<split>.npz`` with ``images`` (N, 3, H, W)
uint8, ``labels`` (N,) int64 (style index), and ``files`` (N,) str (zip member path).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import threading
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from dgm.data.download import download_file
from dgm.training.runs import read_json, write_json
from dgm.utils.paths import data_dir

KAGGLE_DATASET = "steubk/wikiart"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
ARCHIVE_BYTES = 33_662_671_673
SPLITS = ("train", "val", "test")
INDEX_NAME = "classes.csv"
UNCERTAIN = "uncertain artist"


def archive_path() -> Path:
    return data_dir() / "downloads" / "wikiart.zip"


def processed_dir(image_size: int) -> Path:
    return data_dir() / "processed" / "wikiart" / f"{image_size}px"


# --- Download --------------------------------------------------------------------


def download() -> Path:
    """Download the zip into ``data/downloads`` (skipped if present). It is not extracted."""
    return download_file(DOWNLOAD_URL, archive_path(), desc="WikiArt zip (33.7 GB)")


# --- Indexing and splitting ---------------------------------------------------------


def read_index(zf: zipfile.ZipFile) -> list[dict]:
    """Rows of ``classes.csv`` as ``{file, style, artist, subset, phash}``.

    Rows whose file is not a member of the zip are skipped (and counted in a printout).
    """
    members = set(zf.namelist())
    with zf.open(INDEX_NAME) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        table = list(reader)
    rows = [
        {
            "file": r["filename"],
            "style": r["filename"].split("/")[0],
            "artist": r["artist"],
            "subset": r["subset"],
            "phash": r["phash"],
        }
        for r in table
        if r["filename"] in members
    ]
    if len(rows) < len(table):
        print(f"skipped {len(table) - len(rows)} rows of {INDEX_NAME} without a file in the zip")
    return rows


def style_names(rows: Sequence[dict]) -> list[str]:
    """Sorted unique style folders; an image's label is the index of its style here."""
    return sorted({r["style"] for r in rows})


def split_rows(
    rows: Sequence[dict], val_fraction: float, seed: int
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Return ``(splits, info)``.

    ``splits`` maps "train" / "val" / "test" to row lists (original order kept).
    ``info`` counts the dropped rows: ``uncertain`` (subset "uncertain artist") and
    ``test_leak`` (official test rows whose phash occurs in an official train row).
    """
    kept = [r for r in rows if r["subset"] != UNCERTAIN]
    official_train = [r for r in kept if r["subset"] == "train"]
    train_hashes = {r["phash"] for r in official_train}
    test = [r for r in kept if r["subset"] == "test" and r["phash"] not in train_hashes]
    n_test = sum(r["subset"] == "test" for r in kept)

    groups = sorted({r["phash"] for r in official_train})
    n_val = round(len(groups) * val_fraction)
    shuffled = np.random.default_rng(seed).permutation(len(groups))
    val_hashes = {groups[i] for i in shuffled[:n_val]}
    splits = {
        "train": [r for r in official_train if r["phash"] not in val_hashes],
        "val": [r for r in official_train if r["phash"] in val_hashes],
        "test": test,
    }
    info = {"uncertain": len(rows) - len(kept), "test_leak": n_test - len(test)}
    return splits, info


# --- Preprocessing ------------------------------------------------------------------


def load_image(data: bytes, size: int) -> np.ndarray:
    """RGB, center-crop to a square, resize to ``size x size``; uint8 array ``[3, size, size]``."""
    with Image.open(io.BytesIO(data)) as img:
        img.draft("RGB", (2 * size, 2 * size))  # fast JPEG decode at reduced scale (>= 2x size)
        square = ImageOps.fit(img.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
        return np.asarray(square, dtype=np.uint8).transpose(2, 0, 1)


def load_images(archive: Path, files: Sequence[str], size: int, desc: str = "") -> np.ndarray:
    """Decode zip members in threads; each thread opens its own ZipFile (not thread-safe)."""
    local = threading.local()
    opened: list[zipfile.ZipFile] = []
    lock = threading.Lock()

    def work(name: str) -> np.ndarray:
        if not hasattr(local, "zf"):
            local.zf = zipfile.ZipFile(archive)
            with lock:
                opened.append(local.zf)
        return load_image(local.zf.read(name), size)

    try:
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
            images = list(tqdm(pool.map(work, files), total=len(files), desc=desc, unit="img"))
    finally:
        for zf in opened:
            zf.close()
    return np.stack(images) if images else np.zeros((0, 3, size, size), dtype=np.uint8)


def prepare(
    image_size: int = 64, *, val_fraction: float = 0.05, seed: int = 0, force: bool = False
) -> Path:
    """Build ``<split>.npz`` for train/val/test at one resolution (skipped if present)."""
    out = processed_dir(image_size)
    if not force and all((out / f"{split}.npz").exists() for split in SPLITS):
        return out
    archive = archive_path()
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} is missing. Download it first:\n"
            f"    uv run python -m dgm.data.wikiart --sizes {image_size}"
        )
    with zipfile.ZipFile(archive) as zf:
        rows = read_index(zf)
    styles = style_names(rows)
    splits, info = split_rows(rows, val_fraction, seed)
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "image_size": image_size,
        "preprocessing": "RGB, center crop to square, Lanczos resize, uint8, channels first",
        "split": (
            "official subset; val carved from official train by phash group "
            f"(val_fraction={val_fraction}, seed={seed}); "
            f"test rows with a phash in train removed: {info['test_leak']}; "
            f"'{UNCERTAIN}' rows dropped: {info['uncertain']}"
        ),
        "styles": styles,
        "counts": {},
    }
    for split, items in splits.items():
        files = [r["file"] for r in items]
        labels = np.asarray([styles.index(r["style"]) for r in items], dtype=np.int64)
        images = load_images(archive, files, image_size, desc=f"{split} {image_size}px")
        np.savez(out / f"{split}.npz", images=images, labels=labels, files=np.asarray(files))
        counts = {style: int((labels == i).sum()) for i, style in enumerate(styles)}
        meta["counts"][split] = {**counts, "total": len(items)}
    write_json(out / "meta.json", meta)
    return out


# --- PyTorch dataset ------------------------------------------------------------------


class WikiArt(Dataset):
    """A prepared split. Items are ``(image, label)`` with image ``[3, H, W]`` float in [0, 1].

    ``styles`` (optional) keeps only images of those style folders; labels stay global.
    """

    def __init__(self, split: str, image_size: int = 64, styles: Sequence[str] = ()) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        folder = processed_dir(image_size)
        path = folder / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Prepare the data first:\n"
                f"    uv run python -m dgm.data.wikiart --sizes {image_size}"
            )
        self.style_names: list[str] = read_json(folder / "meta.json")["styles"]
        with np.load(path) as data:
            images, labels, files = data["images"], data["labels"], data["files"]
        if styles:
            unknown = sorted(set(styles) - set(self.style_names))
            if unknown:
                raise ValueError(f"unknown styles {unknown}; available: {self.style_names}")
            keep = np.isin(labels, [self.style_names.index(s) for s in styles])
            images, labels, files = images[keep], labels[keep], files[keep]
        self.images = torch.from_numpy(images)  # uint8 [N, 3, H, W]
        self.labels = torch.from_numpy(labels)
        self.files: list[str] = files.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index].float().div_(255), self.labels[index]

    def as_float(self) -> torch.Tensor:
        """The whole split as one ``[N, 3, H, W]`` float tensor in [0, 1]."""
        return self.images.float().div_(255)


def describe(image_size: int) -> str:
    meta = read_json(processed_dir(image_size) / "meta.json")
    lines = [f"WikiArt {image_size}px -> {processed_dir(image_size)}"]
    for split, counts in meta["counts"].items():
        per_style = {k: v for k, v in counts.items() if k != "total"}
        top = sorted(per_style, key=per_style.get, reverse=True)[:3]
        largest = ", ".join(f"{style} {per_style[style]}" for style in top)
        lines.append(f"  {split:<5} {counts['total']:>6} images (largest: {largest})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the Kaggle WikiArt dataset.")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64], help="square image sizes to cache"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.05, help="fraction of phash groups for validation"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the train/val split")
    parser.add_argument("--force", action="store_true", help="rebuild the processed arrays")
    parser.add_argument(
        "--delete-archive",
        action="store_true",
        help="delete the 33.7 GB zip after preparing (other sizes then need a new download)",
    )
    args = parser.parse_args(argv)

    print(f"zip: {download()}")
    for size in args.sizes:
        prepare(size, val_fraction=args.val_fraction, seed=args.seed, force=args.force)
        print(describe(size))
    if args.delete_archive:
        archive_path().unlink()
        print("deleted the zip; preparing another size needs a new 33.7 GB download")


if __name__ == "__main__":
    main()
