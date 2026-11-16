import io
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from dgm.data import wikiart
from dgm.data.synthetic import make_fake_wikiart
from dgm.data.wikiart import WikiArt, load_image, prepare, read_index, split_rows, style_names
from dgm.training.runs import read_json


@pytest.fixture
def fake_index(tmp_path):
    archive = make_fake_wikiart(tmp_path / "fake.zip")
    with zipfile.ZipFile(archive) as zf:
        return zf.namelist(), read_index(zf)


def test_read_index_uses_csv_rows_and_skips_missing_files(fake_index):
    names, rows = fake_index
    files = [r["file"] for r in rows]
    assert len(files) == len(set(files)) == 24 + 8 + 2  # the missing file is skipped
    assert len(names) == len(files) + 1 + 2  # extra style copy and the two csv files
    assert all(r["style"] == r["file"].split("/")[0] for r in rows)
    assert style_names(rows) == ["Cubism", "Impressionism", "Ukiyo_e"]


def test_split_rows(fake_index):
    _, rows = fake_index
    splits, info = split_rows(rows, val_fraction=0.25, seed=0)
    assert info == {"uncertain": 2, "test_leak": 1}
    assert all(r["subset"] != "uncertain artist" for s in splits.values() for r in s)
    files = {name: {r["file"] for r in s} for name, s in splits.items()}
    assert not files["train"] & files["val"]
    assert not files["train"] & files["test"] and not files["val"] & files["test"]
    assert len(splits["test"]) == 7
    train_val = splits["train"] + splits["val"]
    assert not {r["phash"] for r in splits["test"]} & {r["phash"] for r in train_val}
    # the two train rows sharing a phash land in the same split
    counts: dict[str, list[str]] = {}
    for r in train_val:
        counts.setdefault(r["phash"], []).append(r["file"])
    (pair,) = [fs for fs in counts.values() if len(fs) == 2]
    assert (pair[0] in files["train"] and pair[1] in files["train"]) or set(pair) <= files["val"]
    n_groups = len(counts)
    assert n_groups == 23
    assert len({r["phash"] for r in splits["val"]}) == round(n_groups * 0.25)
    again, _ = split_rows(rows, val_fraction=0.25, seed=0)
    assert again == splits


def test_prepare_and_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path))
    make_fake_wikiart(wikiart.archive_path())
    out = prepare(32, val_fraction=0.25, seed=0)

    meta = read_json(out / "meta.json")
    assert meta["styles"] == ["Cubism", "Impressionism", "Ukiyo_e"]
    for split in wikiart.SPLITS:
        with np.load(out / f"{split}.npz") as data:
            assert data["images"].dtype == np.uint8 and data["images"].shape[1:] == (3, 32, 32)
            assert len(data["images"]) == len(data["labels"]) == len(data["files"])
            assert len(data["images"]) == meta["counts"][split]["total"]
            assert data["labels"].min() >= 0 and data["labels"].max() < 3
    assert meta["counts"]["test"]["total"] == 7

    dataset = WikiArt("val", image_size=32)
    image, label = dataset[0]
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert 0 <= label.item() < 3
    assert dataset.as_float().shape == (len(dataset), 3, 32, 32)
    assert "WikiArt 32px" in wikiart.describe(32)


def test_styles_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path))
    make_fake_wikiart(wikiart.archive_path())
    prepare(32, val_fraction=0.25)
    full = WikiArt("train", image_size=32)
    cubism = WikiArt("train", image_size=32, styles=["Cubism"])
    assert 0 < len(cubism) < len(full)
    assert set(cubism.labels.tolist()) == {full.style_names.index("Cubism")}
    with pytest.raises(ValueError, match="Nope"):
        WikiArt("train", image_size=32, styles=["Nope"])


def test_dataset_explains_missing_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"dgm\.data\.wikiart"):
        WikiArt("train", image_size=64)
    with pytest.raises(FileNotFoundError, match=r"dgm\.data\.wikiart"):
        prepare(64)


def test_load_image_crops_non_square_jpeg():
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((70, 120, 3), dtype=np.uint8)).save(buffer, format="JPEG")
    out = load_image(buffer.getvalue(), 24)
    assert out.shape == (3, 24, 24) and out.dtype == np.uint8
