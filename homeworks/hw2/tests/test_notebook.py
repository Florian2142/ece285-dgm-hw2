"""Run both HW2 notebooks top to bottom on a tiny synthetic WikiArt (fewer steps).

While a TODO is open, its cells print "TODO: ..." instead of failing; afterwards they run
your code. Either way no cell may raise.
"""

import json
from pathlib import Path

import matplotlib
import pytest
from IPython.core.interactiveshell import InteractiveShell

from dgm.data.synthetic import make_fake_wikiart
from dgm.data.wikiart import archive_path, prepare

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"
# Parts II-IV of notebook 02 run as their own smoke test (QUICK = True: a few batches per run).
FEWER_STEPS = {
    "STEPS = 3000": "STEPS = 30",
    "ROUNDS = 300": "ROUNDS = 3",
    "OVERFIT_ROUNDS = 1000": "OVERFIT_ROUNDS = 40",
    "QUICK = False": "QUICK = True",
    "TENSORBOARD = True": "TENSORBOARD = False",  # never start a TensorBoard from a test
}


def run_notebook(path: Path) -> list[str]:
    """Execute every code cell in one IPython shell; return the ids of the cells that failed."""
    shell = InteractiveShell.instance()
    failed = []
    try:
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            for old, new in FEWER_STEPS.items():
                source = source.replace(old, new)
            if not shell.run_cell(source, silent=True).success:
                failed.append(cell["id"])
    finally:
        InteractiveShell.clear_instance()
    return failed


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DGM_RUNS_DIR", str(tmp_path / "runs"))  # notebook runs never touch runs/
    make_fake_wikiart(archive_path(), counts={"train": 40, "test": 12})
    prepare(64, val_fraction=0.25)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["01_explore_data.ipynb", "02_gan_step_by_step.ipynb"])
def test_notebook_runs_top_to_bottom(fake_data, name):
    assert run_notebook(NOTEBOOKS / name) == []
