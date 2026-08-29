from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(os.environ.get("POLYDIM_PROJECT", ROOT)).resolve()
PY_FILE = PROJECT / "polydim_v78_monolito.py"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def load_polydim():
    if not PY_FILE.exists():
        pytest.skip(f"POLYDIM Python source not found: {PY_FILE}")
    spec = importlib.util.spec_from_file_location("polydim_target", PY_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PY_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def polydim():
    return load_polydim()


@pytest.fixture
def rng():
    return __import__("numpy").random.default_rng(123456)
