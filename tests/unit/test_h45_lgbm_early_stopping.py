"""H45: LightGBM early stopping + conservative hyperparameters.

The v3 clean model (trained on point-in-time features after the H42 fix) had:
  train_ic = 0.774,  OOS IC = 0.0078  →  99% generalisation failure.

Root cause: n_estimators=500, num_leaves=63, min_child_samples=20 gives the
model ~31,500 leaf nodes — enough to memorise 756k training rows and produce
near-zero OOS IC even when the true signal (live IC ≈ 0.025) exists.

H45 applies two complementary fixes:
  1. LGBM_PARAMS: conservative capacity budget
       n_estimators   500  → 1000  (ceiling; early stopping decides actual count)
       num_leaves      63  →   31  (halved depth)
       min_child_samples 20 →   50  (each split backed by 50+ samples)
       reg_lambda      0.1 →  1.0  (10× stronger L2)
  2. early_stopping(50): halt training when val loss plateaus for 50 rounds.
     This is the mechanically correct fix — the eval_set was wired but never
     consulted because callbacks=[] disabled all callbacks.

Expected on Railway (needs_data_run): train_ic should fall from ~0.77 to
~0.15–0.25; val_ic / OOS IC should close toward the true signal (target ≥
MIN_VAL_IC = 0.05 for model promotion without env pin).

These tests verify LGBM_PARAMS constants and callback wiring via source
inspection — no lightgbm execution required.  The Railway retrain will
confirm whether the IC gap closes.
"""

import inspect
import re

import pytest


# ── LGBM_PARAMS guard tests ───────────────────────────────────────────────────

def test_n_estimators_is_a_budget():
    """n_estimators ≥ 500 — large ceiling so early stopping can find optimal point."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["n_estimators"] >= 500, (
        f"n_estimators={LGBM_PARAMS['n_estimators']}; H45 uses 1000 as a budget — "
        "early_stopping(50) halts training, not a low n_estimators ceiling."
    )


def test_num_leaves_halved():
    """num_leaves ≤ 32 — halved from original 63 to reduce memorisation risk."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["num_leaves"] <= 32, (
        f"num_leaves={LGBM_PARAMS['num_leaves']}; must be ≤ 32 for H45 "
        "(was 63 — allowed complex trees that memorised financial noise)."
    )


def test_min_child_samples_increased():
    """min_child_samples ≥ 40 — was 20; prevents splits on tiny noisy subsets."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["min_child_samples"] >= 40, (
        f"min_child_samples={LGBM_PARAMS['min_child_samples']}; need ≥ 40 "
        "(was 20 — 0.003% of corpus per leaf is memorisation, not generalisation)."
    )


def test_l2_regularisation_increased():
    """reg_lambda ≥ 0.5 — was 0.1; stronger L2 smooths leaf weights."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["reg_lambda"] >= 0.5, (
        f"reg_lambda={LGBM_PARAMS['reg_lambda']}; need ≥ 0.5 for H45 "
        "(was 0.1 — insufficient for high-noise intraday financial data)."
    )


def test_params_dict_has_required_keys():
    """All keys required by LGBMRegressor/LGBMClassifier must be present."""
    from src.models.lgbm import LGBM_PARAMS

    required = {
        "n_estimators", "learning_rate", "num_leaves",
        "min_child_samples", "reg_alpha", "reg_lambda",
        "subsample", "colsample_bytree",
    }
    missing = required - set(LGBM_PARAMS.keys())
    assert not missing, f"LGBM_PARAMS missing keys: {missing}"


# ── Early stopping wiring tests ───────────────────────────────────────────────

def test_callbacks_not_empty_in_source():
    """train() must not pass callbacks=[] — that silently disables early stopping."""
    from src.models import lgbm as lgbm_mod

    src = inspect.getsource(lgbm_mod.LGBMSignalModel.train)
    assert "callbacks=[]" not in src, (
        "callbacks=[] found in LGBMSignalModel.train() — this disables early "
        "stopping even when eval_set is configured. H45 fix: use early_stopping(50)."
    )


def test_early_stopping_imported_in_train():
    """early_stopping must be imported inside train() — confirms the fix is wired."""
    from src.models import lgbm as lgbm_mod

    src = inspect.getsource(lgbm_mod.LGBMSignalModel.train)
    assert "early_stopping" in src, (
        "early_stopping not found in LGBMSignalModel.train() source; "
        "H45 requires callbacks=[early_stopping(50, verbose=False), ...]."
    )


def test_early_stopping_patience_is_sufficient():
    """early_stopping patience ≥ 30 — prevents premature halt on noisy data."""
    from src.models import lgbm as lgbm_mod

    src = inspect.getsource(lgbm_mod.LGBMSignalModel.train)
    m = re.search(r"early_stopping\((\d+)", src)
    assert m, "early_stopping(...) call not found in train() — H45 not applied"
    patience = int(m.group(1))
    assert patience >= 30, (
        f"early_stopping patience={patience} too low; "
        "need ≥ 30 for financial time series (noisy signal causes early exit)."
    )


def test_early_stopping_applied_to_both_heads():
    """Both regressor AND classifier .fit() calls must use early_stopping."""
    from src.models import lgbm as lgbm_mod

    src = inspect.getsource(lgbm_mod.LGBMSignalModel.train)
    # Count occurrences — one per head (regressor + classifier)
    count = src.count("early_stopping(")
    assert count >= 2, (
        f"early_stopping appears {count} time(s) in train(); expected ≥ 2 "
        "(one per head: regressor + classifier)."
    )


# ── Integration test (requires lightgbm) ─────────────────────────────────────

def test_train_uses_lgbm_params_and_early_stopping(monkeypatch):
    """LGBMSignalModel.train() must use LGBM_PARAMS and pass early_stopping callbacks.

    Patches lightgbm constructors to capture kwargs and callback list, then
    verifies the model forwards LGBM_PARAMS and non-empty callbacks.
    Skipped if lightgbm is not installed.
    """
    lgbm = pytest.importorskip("lightgbm")
    import numpy as np

    from src.models.lgbm import LGBM_PARAMS, LGBMSignalModel

    captured_kwargs: list[dict] = []
    captured_callbacks: list[list] = []

    class _FakeModel:
        def __init__(self, **kw):
            captured_kwargs.append(kw)

        def fit(self, X, y, eval_set=None, callbacks=None):
            captured_callbacks.append(callbacks or [])
            self._n = len(y)

        def predict(self, X):
            return np.zeros(len(X))

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

        @property
        def n_features_(self):
            return 2

    monkeypatch.setattr(lgbm, "LGBMRegressor", _FakeModel)
    monkeypatch.setattr(lgbm, "LGBMClassifier", _FakeModel)

    rng = np.random.default_rng(0)
    n = 40
    X = rng.standard_normal((n, 2)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    model = LGBMSignalModel(feature_cols=["a", "b"])
    model.train(X[:30], y[:30], X[30:], y[30:])

    assert len(captured_kwargs) == 2, "Expected 2 model constructions (reg + cls)"
    for kw in captured_kwargs:
        for key, val in LGBM_PARAMS.items():
            assert kw.get(key) == val, (
                f"train() passed {key}={kw.get(key)!r} but LGBM_PARAMS has {val!r}"
            )

    assert len(captured_callbacks) == 2, "Expected 2 .fit() calls"
    for cbs in captured_callbacks:
        assert cbs, "callbacks must be non-empty — empty list disables early stopping"
