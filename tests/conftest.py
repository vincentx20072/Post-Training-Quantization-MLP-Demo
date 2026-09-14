import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKPOINT = ROOT / "artifacts" / "fp32_model.pt"


@pytest.fixture(scope="session")
def trained_model():
    from tinyquant.train import load_model

    if not CHECKPOINT.exists():
        pytest.skip("no trained checkpoint; run `python -m tinyquant.train` first")
    return load_model()
