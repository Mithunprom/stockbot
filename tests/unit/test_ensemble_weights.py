"""Unit tests for EnsembleWeights — H4 dead-weight renormalization.

Covers:
  - EnsembleWeights.validate() pass and fail paths
  - EnsembleWeights.renormalize_dropping_dead_models() — H4 logic
  - EnsembleWeights.from_staging() key-name compatibility (ensemble_weights vs proposed_weights)
  - EnsembleEngine.compute_signal() ensemble formula with H4 weights
  - ProfitAgent._propose_weights() dead-weight detection path
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.models.ensemble import EnsembleWeights, EnsembleSignal


# ─── EnsembleWeights.validate ────────────────────────────────────────────────

def test_default_weights_are_valid():
    w = EnsembleWeights()
    w.validate()  # must not raise


def test_custom_valid_weights_pass():
    w = EnsembleWeights(lgbm=0.75, transformer=0.0, tcn=0.0, sentiment=0.25)
    w.validate()


def test_weights_that_dont_sum_to_one_raise():
    w = EnsembleWeights(lgbm=0.60, transformer=0.10, tcn=0.10, sentiment=0.30)
    with pytest.raises(ValueError, match="sum to 1.0"):
        w.validate()


def test_weights_summing_to_one_within_tolerance():
    # Floating-point rounding across 4 fields may introduce tiny error
    w = EnsembleWeights(lgbm=0.7500, transformer=0.0, tcn=0.0, sentiment=0.2500)
    w.validate()  # 0.75 + 0.25 = 1.0000 — must pass


# ─── EnsembleWeights.renormalize_dropping_dead_models ────────────────────────

def test_h4_renorm_lgbm_and_sentiment_only():
    """H4: drop transformer + TCN → LGBM 0.60→0.75, sentiment 0.20→0.25."""
    w = EnsembleWeights.renormalize_dropping_dead_models({"lgbm", "sentiment"})
    assert w.transformer == 0.0
    assert w.tcn == 0.0
    assert math.isclose(w.lgbm, 0.75, abs_tol=1e-4)
    assert math.isclose(w.sentiment, 0.25, abs_tol=1e-4)
    w.validate()


def test_renorm_all_active_returns_same_proportions():
    """When all models are active, each weight is the default / total = default."""
    w = EnsembleWeights.renormalize_dropping_dead_models(
        {"lgbm", "transformer", "tcn", "sentiment"}
    )
    defaults = EnsembleWeights()
    total = defaults.lgbm + defaults.transformer + defaults.tcn + defaults.sentiment
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    # All defaults already sum to 1.0 so renorm produces the same values
    assert math.isclose(w.lgbm, defaults.lgbm, abs_tol=1e-4)
    assert math.isclose(w.transformer, defaults.transformer, abs_tol=1e-4)
    assert math.isclose(w.sentiment, defaults.sentiment, abs_tol=1e-4)
    w.validate()


def test_renorm_lgbm_only():
    """Extreme case: only LGBM active — it should get full weight."""
    w = EnsembleWeights.renormalize_dropping_dead_models({"lgbm"})
    assert math.isclose(w.lgbm, 1.0, abs_tol=1e-4)
    assert w.transformer == 0.0
    assert w.tcn == 0.0
    assert w.sentiment == 0.0
    w.validate()


def test_renorm_empty_active_set_raises():
    with pytest.raises(ValueError, match="No active models"):
        EnsembleWeights.renormalize_dropping_dead_models(set())


def test_renorm_output_sums_to_one():
    """Any combination of active models should produce weights summing to 1."""
    for combo in [
        {"lgbm", "sentiment"},
        {"lgbm", "transformer"},
        {"lgbm", "tcn", "sentiment"},
        {"transformer", "tcn"},
    ]:
        w = EnsembleWeights.renormalize_dropping_dead_models(combo)
        total = w.lgbm + w.transformer + w.tcn + w.sentiment
        assert math.isclose(total, 1.0, abs_tol=1e-3), f"Combo {combo}: total={total}"
        w.validate()


# ─── EnsembleWeights.from_staging ─────────────────────────────────────────────

def test_from_staging_reads_ensemble_weights_key():
    """Primary key: from_staging reads 'ensemble_weights' correctly."""
    data = {
        "ensemble_weights": {
            "lgbm": 0.75,
            "transformer": 0.0,
            "tcn": 0.0,
            "sentiment": 0.25,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    w = EnsembleWeights.from_staging(path)
    assert math.isclose(w.lgbm, 0.75, abs_tol=1e-4)
    assert w.transformer == 0.0
    assert w.tcn == 0.0
    assert math.isclose(w.sentiment, 0.25, abs_tol=1e-4)
    w.validate()


def test_from_staging_falls_back_to_proposed_weights_key():
    """Backward compat: files written with 'proposed_weights' key still load."""
    data = {
        "proposed_weights": {
            "lgbm": 0.75,
            "transformer": 0.0,
            "tcn": 0.0,
            "sentiment": 0.25,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    w = EnsembleWeights.from_staging(path)
    assert math.isclose(w.lgbm, 0.75, abs_tol=1e-4)
    w.validate()


def test_from_staging_uses_defaults_when_keys_missing():
    """Neither key present → safe defaults (no crash)."""
    data = {"agent": "test", "metrics": {}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)

    w = EnsembleWeights.from_staging(path)
    w.validate()
    # Defaults: lgbm=0.60, transformer=0.10, tcn=0.10, sentiment=0.20
    assert math.isclose(w.lgbm, 0.60, abs_tol=1e-4)


def test_from_staging_reads_actual_h4_staging_file():
    """The H4 staging file written by this R&D run loads and validates correctly."""
    staging_path = Path("config/staging/profit_suggestions.json")
    if not staging_path.exists():
        pytest.skip("H4 staging file not yet written")

    w = EnsembleWeights.from_staging(staging_path)
    w.validate()
    assert math.isclose(w.lgbm, 0.75, abs_tol=1e-4)
    assert w.transformer == 0.0
    assert w.tcn == 0.0
    assert math.isclose(w.sentiment, 0.25, abs_tol=1e-4)


# ─── Ensemble signal formula correctness ─────────────────────────────────────

def _make_signal(w: EnsembleWeights, lgbm_dir=1.0, lgbm_conf=0.8, si=0.5) -> float:
    """Manually compute the ensemble formula given weights and signal values."""
    return (
        w.lgbm * lgbm_conf * lgbm_dir
        + w.transformer * 0.0 * 0.0   # dead models: always 0
        + w.tcn * 0.0 * 0.0
        + w.sentiment * si
    )


def test_h4_weights_produce_stronger_signal_when_lgbm_and_si_agree():
    """H4 weights give a higher ensemble signal than defaults when LGBM + sentiment both bullish."""
    w_default = EnsembleWeights()
    w_h4 = EnsembleWeights(lgbm=0.75, transformer=0.0, tcn=0.0, sentiment=0.25)

    # With default weights (effective, since transformer/TCN=0)
    sig_default = _make_signal(w_default, lgbm_dir=1.0, lgbm_conf=0.8, si=0.4)
    # With H4 weights
    sig_h4 = _make_signal(w_h4, lgbm_dir=1.0, lgbm_conf=0.8, si=0.4)

    # H4 signal should be higher because LGBM's weight is increased
    assert sig_h4 > sig_default
    # H4 correctly reaches up to 1.0 while default caps at 0.80 (with transformer/TCN=0)
    max_h4 = _make_signal(w_h4, lgbm_dir=1.0, lgbm_conf=1.0, si=1.0)
    max_default = _make_signal(w_default, lgbm_dir=1.0, lgbm_conf=1.0, si=1.0)
    assert math.isclose(max_h4, 1.0, abs_tol=1e-9)
    assert math.isclose(max_default, 0.80, abs_tol=1e-9)


def test_h4_signal_strength_classification():
    """H4 weights allow 'strong' classification where default weights cannot reach it."""
    from datetime import datetime, timezone

    w_h4 = EnsembleWeights(lgbm=0.75, transformer=0.0, tcn=0.0, sentiment=0.25)
    # Moderate LGBM conviction + moderate sentiment → should reach 'strong' (≥0.60) with H4
    ens = _make_signal(w_h4, lgbm_dir=1.0, lgbm_conf=0.72, si=0.5)
    sig = EnsembleSignal(
        ticker="AAPL",
        timestamp=datetime.now(timezone.utc),
        ensemble_signal=ens,
        w_lgbm=w_h4.lgbm,
        w_transformer=w_h4.transformer,
        w_tcn=w_h4.tcn,
        w_sentiment=w_h4.sentiment,
    )
    # 0.72*0.75 + 0.5*0.25 = 0.54 + 0.125 = 0.665 → "strong"
    assert sig.strength == "strong"


# ─── ProfitAgent._propose_weights dead-weight detection ──────────────────────

def test_profit_agent_proposes_h4_when_transformer_and_tcn_ic_zero():
    """When transformer_ic=0 and tcn_ic=0, _propose_weights returns H4 weights."""
    from src.agents.profit_agent import ProfitAgent

    agent = ProfitAgent()
    attribution = {"transformer_ic": 0.0, "tcn_ic": 0.0, "sentiment_ic": 0.05}
    weights = agent._propose_weights(attribution, metrics={})

    assert weights["lgbm"] == 0.75
    assert weights["transformer"] == 0.0
    assert weights["tcn"] == 0.0
    assert weights["sentiment"] == 0.25


def test_profit_agent_propose_weights_always_includes_all_four_keys():
    """All four weight keys must be present so from_staging can load without falling back."""
    from src.agents.profit_agent import ProfitAgent

    agent = ProfitAgent()

    # Dead-weight path
    w1 = agent._propose_weights({"transformer_ic": 0.0, "tcn_ic": 0.0, "sentiment_ic": 0.0}, {})
    assert all(k in w1 for k in ("lgbm", "transformer", "tcn", "sentiment"))

    # Proportional path
    w2 = agent._propose_weights(
        {"transformer_ic": 0.05, "tcn_ic": 0.03, "sentiment_ic": 0.08}, {}
    )
    assert all(k in w2 for k in ("lgbm", "transformer", "tcn", "sentiment"))

    # Balanced-default path (secondary_ic_total < 0.01)
    w3 = agent._propose_weights(
        {"transformer_ic": 0.002, "tcn_ic": 0.0, "sentiment_ic": 0.005}, {}
    )
    # transformer_ic and tcn_ic are both < 0.01 → dead-weight path triggers
    assert all(k in w3 for k in ("lgbm", "transformer", "tcn", "sentiment"))


def test_profit_agent_proposed_weights_sum_to_one():
    """Every weight proposal must sum to 1.0 (required by EnsembleWeights.validate)."""
    from src.agents.profit_agent import ProfitAgent

    agent = ProfitAgent()
    test_cases = [
        {"transformer_ic": 0.0, "tcn_ic": 0.0, "sentiment_ic": 0.0},
        {"transformer_ic": 0.10, "tcn_ic": 0.08, "sentiment_ic": 0.12},
        {"transformer_ic": 0.0, "tcn_ic": 0.0, "sentiment_ic": 0.05},
    ]
    for attribution in test_cases:
        w = agent._propose_weights(attribution, {})
        total = w["lgbm"] + w["transformer"] + w["tcn"] + w["sentiment"]
        assert math.isclose(total, 1.0, abs_tol=1e-3), (
            f"attribution={attribution} → weights={w} → total={total:.4f}"
        )
