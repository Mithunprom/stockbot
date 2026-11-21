"""H43: LightGBM regularisation guards — prevent the 99% generalisation failure.

The v3 clean model (trained on point-in-time features after the H42 fix) had:
  train_ic = 0.774,  OOS IC = 0.0078  →  99% generalisation failure.

Root cause: n_estimators=500, num_leaves=63, min_child_samples=20 gives the
model ~31 500 leaf nodes — enough to memorise 756k training rows and produce
near-zero OOS IC even when the true signal (live IC ≈ 0.025) exists.

H43 replaces those defaults with a conservative budget:
  n_estimators    500  →  100   (5× fewer trees)
  num_leaves       63  →   15   (shallower trees)
  min_child_samples 20 →  200   (each split backed by 200+ samples)
  reg_alpha        0.1 →  1.0   (10× stronger L1)
  reg_lambda       0.1 →  1.0   (10× stronger L2)

Expected on Railway (needs_data_run): train_ic should fall from ~0.77 to
~0.10–0.20; val_ic / OOS IC should close toward the true signal (target ≥
MIN_VAL_IC = 0.05 for model promotion).

These tests verify only the LGBM_PARAMS constants (no lightgbm execution
required).  The numeric bounds are the minimum conservative budget for H43;
the Railway retrain will confirm whether the IC gap actually closes.
"""

import pytest


def test_n_estimators_is_conservative():
    """≤ 150 trees — limits total leaf-node capacity on the training corpus."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["n_estimators"] <= 150, (
        f"n_estimators={LGBM_PARAMS['n_estimators']} is too high; "
        "500 trees memorised the training set (train_ic 0.774)."
    )


def test_num_leaves_is_shallow():
    """≤ 31 leaves — prevents the model from carving out noise pockets."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["num_leaves"] <= 31, (
        f"num_leaves={LGBM_PARAMS['num_leaves']} still too deep; "
        "target ≤ 31 (depth ~5) for 15-bar intraday prediction."
    )


def test_min_child_samples_requires_density():
    """≥ 100 samples per leaf — with 756k training rows each split is real.

    At min_child_samples=20 a leaf can represent just 0.003% of the training
    corpus — essentially memorising individual ticker-day patterns.
    """
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["min_child_samples"] >= 100, (
        f"min_child_samples={LGBM_PARAMS['min_child_samples']} is too low; "
        "need ≥ 100 to prevent overfit on financial time series."
    )


def test_l1_regularisation_is_meaningful():
    """reg_alpha ≥ 0.5 — sparse L1 penalty shrinks noisy features toward zero."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["reg_alpha"] >= 0.5, (
        f"reg_alpha={LGBM_PARAMS['reg_alpha']}; need ≥ 0.5 for H43."
    )


def test_l2_regularisation_is_meaningful():
    """reg_lambda ≥ 0.5 — L2 smooths leaf weights against noisy outcomes."""
    from src.models.lgbm import LGBM_PARAMS

    assert LGBM_PARAMS["reg_lambda"] >= 0.5, (
        f"reg_lambda={LGBM_PARAMS['reg_lambda']}; need ≥ 0.5 for H43."
    )


def test_params_dict_is_complete():
    """All keys required by LGBMRegressor/LGBMClassifier must be present."""
    from src.models.lgbm import LGBM_PARAMS

    required = {
        "n_estimators", "learning_rate", "num_leaves",
        "min_child_samples", "reg_alpha", "reg_lambda",
        "subsample", "colsample_bytree",
    }
    missing = required - set(LGBM_PARAMS.keys())
    assert not missing, f"LGBM_PARAMS missing keys: {missing}"


def test_capacity_budget():
    """Total leaf-node count (n_estimators × num_leaves) must be ≤ 3 000.

    The old config: 500 × 63 = 31 500 — enough to memorise 756k rows.
    The new budget: 100 × 15 = 1 500 — forces general patterns only.
    """
    from src.models.lgbm import LGBM_PARAMS

    capacity = LGBM_PARAMS["n_estimators"] * LGBM_PARAMS["num_leaves"]
    assert capacity <= 3_000, (
        f"Total capacity={capacity} leaf-nodes is too high for 756k training rows; "
        "keep n_estimators × num_leaves ≤ 3 000."
    )


def test_train_method_uses_lgbm_params(monkeypatch):
    """LGBMSignalModel.train() must pass LGBM_PARAMS to the underlying models.

    Patches the lightgbm constructors to capture the kwargs, then verifies
    the model forwards LGBM_PARAMS without substituting local overrides.
    Skipped if lightgbm is not installed in the current environment.
    """
    lgbm = pytest.importorskip("lightgbm")
    import numpy as np

    from src.models.lgbm import LGBM_PARAMS, LGBMSignalModel

    captured: list[dict] = []

    class _FakeModel:
        def __init__(self, **kw):
            captured.append(kw)

        def fit(self, X, y, **kw):
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

    assert len(captured) == 2, "Expected exactly 2 model constructions (reg + cls)"
    for kw in captured:
        for key, val in LGBM_PARAMS.items():
            assert kw.get(key) == val, (
                f"train() passed {key}={kw.get(key)!r} but LGBM_PARAMS has {val!r}; "
                "the method must use the module-level LGBM_PARAMS constant."
            )
