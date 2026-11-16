"""Test helpers: tiny synthetic datasets that mimic the real folder layouts."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from dgm.data.chest_xray import CLASSES

DEFAULT_COUNTS = {"train": 24, "val": 4, "test": 8}
_PATIENT_OFFSET = {"train": 0, "val": 1000, "test": 2000}


def make_fake_chest_xray(
    root: Path,
    counts: Mapping[str, int] = DEFAULT_COUNTS,
    image_hw: tuple[int, int] = (72, 90),
    seed: int = 0,
) -> Path:
    """Write random grayscale JPEGs in the Kaggle layout under ``root``; return the split root.

    Mirrors the real archive: data nested in ``chest_xray/chest_xray/``, a ``__MACOSX``
    folder, Kaggle file names (``person7_bacteria_13.jpeg``, ``IM-0007-0001.jpeg``), and
    two images per patient so that patient-level splitting is exercised.
    """
    rng = np.random.default_rng(seed)
    split_root = root / "chest_xray" / "chest_xray"
    for split, n in counts.items():
        for i in range(n):
            label = i % 2
            patient = _PATIENT_OFFSET[split] + i // 4  # 2 images per patient and class
            name = (
                f"person{patient}_bacteria_{i}.jpeg" if label else f"IM-{patient:04d}-{i:04d}.jpeg"
            )
            path = split_root / split / CLASSES[label] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, size=image_hw, dtype=np.uint8)).save(
                path, quality=90
            )
    (root / "chest_xray" / "__MACOSX" / "chest_xray" / "train").mkdir(parents=True, exist_ok=True)
    return split_root


_WIKIART_COLUMNS = [
    "filename",
    "artist",
    "genre",
    "description",
    "phash",
    "width",
    "height",
    "genre_count",
    "subset",
]


def make_fake_wikiart(
    archive: Path,
    counts: Mapping[str, int] = {"train": 24, "test": 8},
    styles: Sequence[str] = ("Impressionism", "Cubism", "Ukiyo_e"),
    image_hw: tuple[int, int] = (72, 90),
    seed: int = 0,
) -> Path:
    """Write a small zip in the Kaggle WikiArt layout at ``archive``; return ``archive``.

    Members are ``<Style>/<artist>_<title>-<i>.jpg`` plus ``classes.csv`` and an unused
    ``wclasses.csv``. Built-in edge cases: 2 "uncertain artist" rows; one train image that
    also exists as a copy in a second style folder (zip member, no csv row); one test row
    with the phash of a train row (leak); two train rows sharing a phash; one csv row
    whose file is missing from the zip.
    """
    rng = np.random.default_rng(seed)
    height, width = image_hw

    def jpeg() -> bytes:
        buffer = io.BytesIO()
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    members: dict[str, bytes] = {}
    rows: list[dict[str, str]] = []

    def add(subset: str, i: int, *, write: bool = True) -> dict[str, str]:
        style = styles[i % len(styles)]
        name = f"{style}/artist-{i}_title-{subset.split()[0]}-{i}.jpg"
        if write:
            members[name] = jpeg()
        row = {
            "filename": name,
            "artist": f"artist-{i}",
            "genre": f"['{style.replace('_', ' ')}']",
            "description": "",
            "phash": f"{int(rng.integers(0, 1 << 62)):016x}",
            "width": str(width),
            "height": str(height),
            "genre_count": "1",
            "subset": subset,
        }
        rows.append(row)
        return row

    train = [add("train", i) for i in range(counts.get("train", 0))]
    test = [add("test", 1000 + i) for i in range(counts.get("test", 0))]
    if len(train) >= 3 and test:
        train[2]["phash"] = train[1]["phash"]  # (d) phash pair, must stay together
        test[0]["phash"] = train[0]["phash"]  # (c) leaked test row
    if train:  # (b) extra copy in a second style folder, zip only
        first = train[0]["filename"]
        other = next(s for s in styles if s != first.split("/")[0])
        members[f"{other}/{first.split('/')[1]}"] = members[first]
        train[0]["genre_count"] = "2"
    add("uncertain artist", 2000)  # (a)
    add("uncertain artist", 2001)
    add("train", 3000, write=False)  # (e) listed, but not in the zip

    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=_WIKIART_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        zf.writestr("classes.csv", table.getvalue())
        zf.writestr("wclasses.csv", "filename,artist\n")  # (f) unused
    return archive
