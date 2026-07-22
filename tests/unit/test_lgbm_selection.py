"""Checkpoint selection must let a retrained model win.

Selecting by the IC embedded in the filename meant the best score ever
recorded ruled forever — the May 28 model (0.1860) beat every fresher
checkpoint even after its live IC decayed to zero (2026-07-22).
"""

from __future__ import annotations

import json

import pytest

from src.models import lgbm as lgbm_mod
from src.models.lgbm import LGBMSignalModel


def _write(tmp_path, name, val_ic, trained_at=None):
    (tmp_path / f"{name}.pkl").write_bytes(b"stub")
    meta = {"val_ic": val_ic, "feature_cols": []}
    if trained_at:
        meta["trained_at"] = trained_at
    (tmp_path / f"{name}.json").write_text(json.dumps(meta))


def test_fresher_model_beats_higher_historical_ic(tmp_path, monkeypatch):
    monkeypatch.setattr(lgbm_mod, "MODEL_DIR", tmp_path)
    _write(tmp_path, "lgbm_ic_0.1860", 0.1860, "2026-05-28T00:00:00+00:00")
    _write(tmp_path, "lgbm_ic_0.1654", 0.1654, "2026-07-22T00:00:00+00:00")
    assert LGBMSignalModel._select_checkpoint().name == "lgbm_ic_0.1654.pkl"


def test_low_ic_retrain_cannot_take_over(tmp_path, monkeypatch):
    # A garbage retrain must not win on recency alone
    monkeypatch.setattr(lgbm_mod, "MODEL_DIR", tmp_path)
    _write(tmp_path, "lgbm_ic_0.1860", 0.1860, "2026-05-28T00:00:00+00:00")
    _write(tmp_path, "lgbm_ic_0.0100", 0.0100, "2026-07-22T00:00:00+00:00")
    assert LGBMSignalModel._select_checkpoint().name == "lgbm_ic_0.1860.pkl"


def test_env_pin_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(lgbm_mod, "MODEL_DIR", tmp_path)
    _write(tmp_path, "lgbm_ic_0.1860", 0.1860, "2026-05-28T00:00:00+00:00")
    _write(tmp_path, "lgbm_ic_0.1654", 0.1654, "2026-07-22T00:00:00+00:00")
    monkeypatch.setenv("LGBM_MODEL_PATH", str(tmp_path / "lgbm_ic_0.1860.pkl"))
    assert LGBMSignalModel._select_checkpoint().name == "lgbm_ic_0.1860.pkl"


def test_no_checkpoints_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(lgbm_mod, "MODEL_DIR", tmp_path)
    monkeypatch.delenv("LGBM_MODEL_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        LGBMSignalModel._select_checkpoint()
