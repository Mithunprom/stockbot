"""Model promotion: a retrained model must be able to win, and a mismatched
model must not be served.

Two independent defects made the promotion path a dead end in 2026:

  1. `save()` never wrote `trained_at`, while `_select_checkpoint` orders by it
     and sorts checkpoints lacking it OLDEST. Every model RetrainAgent produced
     was therefore unpromotable no matter how good it was — nine of the ten
     checkpoints on disk had no such field.
  2. Nothing recorded WHICH feature definitions a model was trained against, so
     a model could be served features whose meaning had changed underneath it.
     Nothing errors when that happens; the predictions simply stop ranking.

Both are pinned here.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.features.indicators import FEATURE_PIPELINE_VERSION
from src.models.lgbm import LGBMSignalModel


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    import src.models.lgbm as lgbm_mod

    monkeypatch.setattr(lgbm_mod, "MODEL_DIR", tmp_path)
    return tmp_path


class _StubEstimator:
    """Picklable stand-in for a fitted LightGBM estimator.

    These tests exercise the SAVE/SELECT/LOAD contract, not LightGBM. Training
    a real booster here also segfaults the suite on macOS: lightgbm and torch
    each vendor their own libomp, and loading both in one process crashes in
    the native layer. Stubbing keeps the tests hermetic and fast.
    """

    def __init__(self, weight: float = 0.01) -> None:
        self.weight = weight

    def predict(self, X):
        return np.asarray(X, dtype=np.float64)[:, 0] * self.weight


def _fitted(val_ic: float = 0.15) -> LGBMSignalModel:
    """A model with fitted-looking internals — enough to pickle and reload."""
    m = LGBMSignalModel(feature_cols=["a", "b", "c"])
    m.regressor = _StubEstimator()
    m.classifier = _StubEstimator(0.5)
    m.train_ic = 0.2
    m.val_ic = val_ic
    m.val_dir_acc = 0.55
    return m


def test_save_stamps_trained_at(model_dir):
    """REGRESSION: without this the model can never be selected by recency."""
    path = _fitted().save()

    meta = json.loads(Path(str(path)[:-4] + ".json").read_text())
    assert meta.get("trained_at"), (
        "save() did not record trained_at — _select_checkpoint sorts such "
        "checkpoints oldest, so this model could never be promoted"
    )

    with open(path, "rb") as f:
        assert pickle.load(f).get("trained_at")


def test_save_stamps_feature_pipeline_version(model_dir):
    path = _fitted().save()
    meta = json.loads(Path(str(path)[:-4] + ".json").read_text())
    assert meta["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION


def test_a_fresh_model_beats_an_older_one(model_dir):
    """The whole point of recency ordering: a retrain must be able to win."""
    stale = model_dir / "lgbm_ic_0.9000.pkl"
    with open(stale, "wb") as f:
        pickle.dump({"regressor": None, "classifier": None, "feature_cols": []}, f)
    Path(str(stale)[:-4] + ".json").write_text(json.dumps({
        "val_ic": 0.90,                       # better score, much older
        "trained_at": "2020-01-01T00:00:00+00:00",
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
    }))

    fresh = _fitted(val_ic=0.12).save()

    assert LGBMSignalModel._select_checkpoint() == fresh, (
        "a stale checkpoint with a flattering filename IC outranked a freshly "
        "trained model — this is the 2026-07-22 staleness trap"
    )


def test_load_refuses_a_pipeline_version_mismatch(model_dir):
    """The guard that makes silent train/serve skew impossible."""
    path = _fitted().save()

    with open(path, "rb") as f:
        data = pickle.load(f)
    data["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION + 1
    with open(path, "wb") as f:
        pickle.dump(data, f)

    with pytest.raises(ValueError, match="Feature pipeline mismatch"):
        LGBMSignalModel.load(path)


def test_legacy_checkpoint_is_treated_as_v1_and_fails_closed(model_dir):
    """A checkpoint predating the version field was trained on v1 definitions.

    Treating it as "unknown, probably fine" is what allowed a v1 model to be
    served v2 features for two months. It must fail closed instead: no LightGBM
    signal means no entries, and not trading is always recoverable.
    """
    path = _fitted().save()

    with open(path, "rb") as f:
        data = pickle.load(f)
    data.pop("feature_pipeline_version")
    with open(path, "wb") as f:
        pickle.dump(data, f)

    if FEATURE_PIPELINE_VERSION == 1:
        LGBMSignalModel.load(path)             # v1 serving v1 — fine
    else:
        with pytest.raises(ValueError, match="Feature pipeline mismatch"):
            LGBMSignalModel.load(path)


def test_load_best_checkpoint_returns_none_rather_than_raising(model_dir):
    """The refusal must degrade to "no signal", not crash the signal loop."""
    from src.models.lgbm import load_best_checkpoint

    path = _fitted().save()
    with open(path, "rb") as f:
        data = pickle.load(f)
    data["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION + 1
    with open(path, "wb") as f:
        pickle.dump(data, f)

    assert load_best_checkpoint() is None


def test_round_trip_preserves_predictions(model_dir):
    m = _fitted()
    path = m.save()
    reloaded = LGBMSignalModel.load(path)

    X = np.random.default_rng(1).normal(size=(10, 3)).astype(np.float32)
    np.testing.assert_allclose(
        m.regressor.predict(X), reloaded.regressor.predict(X), rtol=1e-9,
    )
