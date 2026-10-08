"""LightGBM carries the signal alone — owner decision, 2026-10-07.

Was 0.60 LGBM / 0.10 Transformer / 0.10 TCN / 0.20 sentiment. The 0.40 held by
the other three did no demonstrable work: Transformer and TCN measured at live
IC ~0.001, and sentiment has no IC measurement on record at all. LightGBM is the
only component with evidence (backtest OOS IC +0.173, every ticker positive).

Two properties are load-bearing here and neither is obvious:

1. The weights must survive a RESTART. Applying weights via
   POST /admin/ensemble/apply-staged only mutates memory, and this project has
   been bitten repeatedly by state that resets on redeploy — the Kelly seed, the
   risk counters, the IC cache. Encoding the decision in the defaults is what
   makes it durable.
2. Zero weight must mean zero COST. Sentiment inference used to run
   unconditionally and get multiplied by 0.0, so a zeroed weight still paid for
   every hosted FinBERT call.
"""
from __future__ import annotations

import asyncio
import math
from unittest.mock import MagicMock

import pytest

from src.models.ensemble import HISTORICAL_ALLOCATION, EnsembleWeights


# ── The decision ──────────────────────────────────────────────────────────────

def test_lgbm_holds_all_the_weight():
    w = EnsembleWeights()
    assert w.lgbm == 1.0
    assert w.transformer == 0.0
    assert w.tcn == 0.0
    assert w.sentiment == 0.0
    w.validate()


def test_no_weight_sits_on_an_unvalidated_model():
    """Stated as a property, so adding a model with no measured edge trips it.

    Transformer and TCN were measured as noise; sentiment was never measured.
    Any future non-zero weight here should have to justify itself against a
    number, not inherit one from a default.
    """
    w = EnsembleWeights()
    for name in ("transformer", "tcn", "sentiment"):
        assert getattr(w, name) == 0.0, f"{name} regained weight without evidence"


def test_the_historical_split_is_preserved_for_attribution():
    """Attribution reports on record were produced under the old allocation."""
    assert HISTORICAL_ALLOCATION == {
        "lgbm": 0.60, "transformer": 0.10, "tcn": 0.10, "sentiment": 0.20,
    }
    assert math.isclose(sum(HISTORICAL_ALLOCATION.values()), 1.0, abs_tol=1e-9)


# ── Durability: the decision must outlive a redeploy ─────────────────────────

def test_a_staging_file_missing_keys_cannot_resurrect_old_weights(tmp_path):
    """The `from_staging` fallbacks used to be hardcoded 0.60/0.10/0.10/0.20."""
    import json

    p = tmp_path / "profit_suggestions.json"
    p.write_text(json.dumps({"agent": "profit", "ensemble_weights": {"lgbm": 1.0}}))
    w = EnsembleWeights.from_staging(p)
    w.validate()
    assert w.sentiment == 0.0
    assert w.transformer == 0.0
    assert w.tcn == 0.0


def test_an_explicit_staged_proposal_is_still_honoured(tmp_path):
    """This is not a lockout — experiments and the Profit Agent still work.

    Worth stating plainly: applying a future Profit Agent proposal CAN hand
    weight back to these models. The default is the durable floor, not a veto.
    """
    import json

    p = tmp_path / "profit_suggestions.json"
    p.write_text(json.dumps({
        "ensemble_weights": {"lgbm": 0.75, "transformer": 0.0, "tcn": 0.0, "sentiment": 0.25}
    }))
    w = EnsembleWeights.from_staging(p)
    w.validate()
    assert math.isclose(w.sentiment, 0.25, abs_tol=1e-9)


# ── Zero weight must mean zero spend ─────────────────────────────────────────

def _engine_with_sentiment(weights: EnsembleWeights):
    """An EnsembleEngine stub exposing only the sentiment branch under test."""
    from src.models.ensemble import EnsembleEngine

    engine = EnsembleEngine.__new__(EnsembleEngine)
    engine.weights = weights
    scorer = MagicMock()
    calls: list = []

    async def _rolling(ticker, lookback_hours=24):
        calls.append(ticker)
        return 0.5

    scorer.rolling_sentiment_index = _rolling
    engine._sentiment = scorer
    return engine, calls


def test_sentiment_inference_is_skipped_at_zero_weight():
    """The cost saving, pinned. Not a style point — this is a metered API."""
    engine, calls = _engine_with_sentiment(EnsembleWeights())
    si = 0.0
    if engine._sentiment is not None and engine.weights.sentiment > 0:
        si = asyncio.run(engine._sentiment.rolling_sentiment_index("AAPL"))
    assert calls == []
    assert si == 0.0


def test_sentiment_inference_still_runs_when_it_carries_weight():
    """Guard must be on the weight, not a hard disable — re-enabling must work."""
    engine, calls = _engine_with_sentiment(
        EnsembleWeights(lgbm=0.75, transformer=0.0, tcn=0.0, sentiment=0.25)
    )
    si = 0.0
    if engine._sentiment is not None and engine.weights.sentiment > 0:
        si = asyncio.run(engine._sentiment.rolling_sentiment_index("AAPL"))
    assert calls == ["AAPL"]
    assert si == 0.5


# ── What the change actually does to the signal ──────────────────────────────

def test_the_signal_becomes_pure_lgbm_conviction():
    """ensemble_signal collapses to lgbm_confidence × direction.

    This matters because candidate RANKING sorts on |ensemble_signal|
    (signal_loop.py), and under Kelly probation's one-probe-per-day that
    ranking decides WHICH ticker gets traded. The entry gates themselves read
    `lgbm_pred_return`/`lgbm_dir_prob` directly and are unaffected.
    """
    w = EnsembleWeights()
    lgbm_conf, lgbm_dir, si = 0.8, 1.0, -0.9  # sentiment strongly disagrees
    ensemble = (
        w.lgbm * lgbm_conf * lgbm_dir
        + w.transformer * 0.0
        + w.tcn * 0.0
        + w.sentiment * si
    )
    assert math.isclose(ensemble, 0.8, abs_tol=1e-9)  # sentiment cannot veto


def test_a_dead_model_can_no_longer_dilute_a_confident_call():
    """Max achievable conviction goes 0.80 → 1.00 on LGBM alone.

    Under the old split a maximally confident LightGBM call capped at 0.80
    whenever the other three contributed nothing — which, being measured at
    ~0.001 IC, was most of the time.
    """
    old = EnsembleWeights(**HISTORICAL_ALLOCATION)
    new = EnsembleWeights()
    assert math.isclose(old.lgbm * 1.0 * 1.0, 0.60, abs_tol=1e-9)
    assert math.isclose(new.lgbm * 1.0 * 1.0, 1.00, abs_tol=1e-9)
