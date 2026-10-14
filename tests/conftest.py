from pathlib import Path
import pytest

from toxifusionnet.config import load_config
from toxifusionnet.data.splits import make_splits
from toxifusionnet.data.synthetic import make_synthetic
from toxifusionnet.training.trainer import train_run


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return repository root path dynamically."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def workspace_dir(repo_root: Path) -> Path:
    """Alias for repo_root for workspace-relative path resolution."""
    return repo_root


@pytest.fixture(scope="session")
def synth_df():
    return make_splits(make_synthetic(500, seed=3))


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, synth_df):
    cfg = load_config("configs/smoke_test.yaml", ["train.epochs=2", "train.batch_size=32"])
    run_dir = tmp_path_factory.mktemp("run")
    train_run(cfg, synth_df, run_dir)
    return run_dir


@pytest.fixture(autouse=True)
def _reset_label_taxonomy():
    """Tests may switch the target taxonomy (MultiBully); always restore the default afterwards."""
    yield
    from toxifusionnet.labels import apply_label_config
    apply_label_config(None)
