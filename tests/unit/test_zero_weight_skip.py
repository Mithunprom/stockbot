"""Zero weight means zero work — and must never mean a different number.

v0.9.3 set Transformer and TCN weights to 0.0 (measured live IC ~0.001). The
latency agent then escalated `transformer_inference p95=347ms` against its 200ms
threshold: a breached SLA paid for two models contributing nothing. Their
inference is now guarded on their own weight, as sentiment's already was.

The load-bearing property is NUMERICAL IDENTITY. Each term enters the sum as
`weight * conf * dir`, so a skipped model contributes exactly 0.0 either way.
If skipping could change a signal value this would be a strategy change wearing
a performance-fix label — the thing to be most careful about here.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.models.ensemble import EnsembleEngine, EnsembleWeights


class _Spy:
    """Stands in for a model, recording whether predict() was reached."""

    def __init__(self, dir_=1.0, conf=0.7):
        self.calls = 0
        self._ret = (dir_, conf)

    def predict(self, *args, **kwargs):
        self.calls += 1
        return self._ret


class _Sentiment:
    def __init__(self, value=0.5):
        self.calls = 0
        self._value = value

    async def rolling_sentiment_index(self, ticker, lookback_hours=24):
        self.calls += 1
        return self._value


def _engine(weights: EnsembleWeights):
    """An EnsembleEngine with spies in place of real models.

    Built via __new__ so no checkpoint loading happens; compute_signal only
    touches the attributes set here.
    """
    e = EnsembleEngine.__new__(EnsembleEngine)
    e.weights = weights
    e._lgbm = _Spy(dir_=1.0, conf=0.8)
    e._transformer = _Spy(dir_=1.0, conf=0.6)
    e._tcn = _Spy(dir_=-1.0, conf=0.4)
    e._sentiment = _Sentiment(0.5)
    return e


def _lgbm_quad(dir_=1.0, conf=0.8, pred=0.01, prob=0.8):
    """LightGBM returns a 4-tuple, unlike the other models' 2-tuple."""

    class _L:
        calls = 0

        def predict(self, *a, **k):
            _L.calls += 1
            return (dir_, conf, pred, prob)

    return _L()


def _features(seq=30, n=30):
    return np.random.RandomState(0).randn(seq, n).astype(np.float64)


def _run(engine):
    f = _features()
    return asyncio.run(
        engine.compute_signal(ticker="AAPL", features_1m=f, features_5m=f)
    )


# ── The skip happens ─────────────────────────────────────────────────────────

def test_transformer_and_tcn_are_not_invoked_at_zero_weight():
    """The latency saving, pinned. This is the whole point of the change."""
    e = _engine(EnsembleWeights())          # lgbm 1.0, everything else 0.0
    e._lgbm = _lgbm_quad()
    _run(e)
    assert e._transformer.calls == 0
    assert e._tcn.calls == 0
    assert e._sentiment.calls == 0


def test_a_restored_weight_restores_inference():
    """Guarded on the WEIGHT, not hard-disabled.

    If this ever fails the models are unreachable, and a retrained Transformer
    could never be evaluated — the guard would have become a deletion.
    """
    e = _engine(EnsembleWeights(lgbm=0.4, transformer=0.3, tcn=0.2, sentiment=0.1))
    e._lgbm = _lgbm_quad()
    _run(e)
    assert e._transformer.calls == 1
    assert e._tcn.calls == 1
    assert e._sentiment.calls == 1


@pytest.mark.parametrize("field,kwargs", [
    ("transformer", dict(lgbm=0.5, transformer=0.5, tcn=0.0, sentiment=0.0)),
    ("tcn", dict(lgbm=0.5, transformer=0.0, tcn=0.5, sentiment=0.0)),
    ("sentiment", dict(lgbm=0.5, transformer=0.0, tcn=0.0, sentiment=0.5)),
])
def test_each_model_is_guarded_independently(field, kwargs):
    """One non-zero weight must not drag the others' inference along."""
    e = _engine(EnsembleWeights(**kwargs))
    e._lgbm = _lgbm_quad()
    _run(e)
    ran = {
        "transformer": e._transformer.calls,
        "tcn": e._tcn.calls,
        "sentiment": e._sentiment.calls,
    }
    assert ran[field] == 1, f"{field} should have run"
    for other, n in ran.items():
        if other != field:
            assert n == 0, f"{other} ran despite zero weight"


# ── The number must not move ─────────────────────────────────────────────────

@pytest.mark.parametrize("weights", [
    EnsembleWeights(),                                                   # prod today
    EnsembleWeights(lgbm=0.6, transformer=0.1, tcn=0.1, sentiment=0.2),  # historical
    EnsembleWeights(lgbm=0.75, transformer=0.0, tcn=0.0, sentiment=0.25),
    EnsembleWeights(lgbm=0.4, transformer=0.3, tcn=0.2, sentiment=0.1),
    EnsembleWeights(lgbm=0.0, transformer=0.5, tcn=0.5, sentiment=0.0),
])
def test_ensemble_signal_matches_the_unskipped_computation(weights):
    """Skipping avoids work; it must not alter the result.

    Computes the sum longhand from the spies' own return values — what
    inference WOULD have produced — and requires the engine to agree exactly.
    For a zero-weighted model the term is 0.0 whether or not predict() ran, so
    the two must be bit-identical, not merely close.
    """
    e = _engine(weights)
    e._lgbm = _lgbm_quad(dir_=1.0, conf=0.8)
    sig = _run(e)

    w = weights
    expected = (
        w.lgbm * 0.8 * 1.0
        + w.transformer * (0.6 * 1.0 if w.transformer > 0 else 0.0)
        + w.tcn * (0.4 * -1.0 if w.tcn > 0 else 0.0)
        + w.sentiment * (0.5 if w.sentiment > 0 else 0.0)
    )
    expected = max(-1.0, min(1.0, expected))
    assert sig.ensemble_signal == expected


def test_a_zero_weighted_model_contributes_zero_either_way():
    """The algebraic justification, stated as a test.

    Same engine, same inputs; only difference is whether the dead models'
    weights are zero. The lgbm-only signal must equal the lgbm term alone,
    proving the skipped terms added nothing rather than being unmeasured.
    """
    e = _engine(EnsembleWeights())
    e._lgbm = _lgbm_quad(dir_=1.0, conf=0.8)
    sig = _run(e)
    assert sig.ensemble_signal == pytest.approx(0.8, abs=0.0)


# ── Serialization of skipped models ──────────────────────────────────────────

def test_skipped_models_report_zero_not_none():
    """`/trades` and signal_loop do `float(sig.transformer_confidence)`.

    A None here would raise there. The DB columns are nullable Float, so
    nothing would reject it at write time — it would surface later as a
    TypeError in the exit path or as a null in the ledger.
    """
    e = _engine(EnsembleWeights())
    e._lgbm = _lgbm_quad()
    sig = _run(e)
    for field in ("transformer_confidence", "transformer_direction",
                  "tcn_confidence", "tcn_direction", "sentiment_index"):
        val = getattr(sig, field)
        assert val is not None, f"{field} is None"
        assert isinstance(val, float), f"{field} is {type(val).__name__}"
        assert val == 0.0
    # The fields consumers actually coerce must survive float().
    assert float(sig.transformer_confidence) == 0.0
    assert float(sig.tcn_confidence) == 0.0


def test_weights_are_still_reported_on_the_signal():
    """Attribution needs to know the weights that produced the signal."""
    e = _engine(EnsembleWeights())
    e._lgbm = _lgbm_quad()
    sig = _run(e)
    assert sig.w_lgbm == 1.0
    assert sig.w_transformer == 0.0
    assert sig.w_tcn == 0.0
    assert sig.w_sentiment == 0.0
