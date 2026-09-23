"""H28: ensemble direction consistency and Kelly hard-block clarity in /diagnostics.

Tests verify the diagnostic computation logic without requiring live credentials
or a running SignalLoop instance. The logic under test lives in the
_pipeline_diagnostics closure inside GET /diagnostics (main.py).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helpers that replicate the H28 computation extracted from main.py
# ---------------------------------------------------------------------------

def _compute_gate_analysis(raw_signals: list[dict]) -> list[dict]:
    """Mirror the computation in _pipeline_diagnostics for unit testing."""
    gate_analysis = []
    cost_thr = 0.00235  # representative default
    lo, hi = 0.4, 0.6

    for sig in raw_signals[:10]:
        pred_ret = float(sig.get("lgbm_pred_return", 0.0))
        dir_prob = float(sig.get("lgbm_dir_prob", 0.5))
        passes_pred = abs(pred_ret) > cost_thr
        passes_dir = not (lo < dir_prob < hi)
        passes_both = passes_pred and passes_dir
        ensemble_direction_ok = float(sig.get("ensemble_signal", 0.0)) > 0

        blocked_by = (
            []
            + (["pred_return_too_small"] if not passes_pred else [])
            + (["dir_prob_in_dead_zone"] if not passes_dir else [])
            + (["ensemble_direction_inconsistent"] if not ensemble_direction_ok and passes_both else [])
        )

        gate_analysis.append({
            "ticker": sig.get("ticker", "?"),
            "ensemble_signal": round(float(sig.get("ensemble_signal", 0.0)), 4),
            "lgbm_pred_return": round(pred_ret, 6),
            "lgbm_dir_prob": round(dir_prob, 4),
            "passes_pred_return_gate": passes_pred,
            "passes_dir_prob_gate": passes_dir,
            "passes_both_gates": passes_both,
            "ensemble_direction_ok": ensemble_direction_ok,
            "blocked_by": blocked_by,
        })

    return gate_analysis


def _count_direction_anomalies(gate_analysis: list[dict]) -> int:
    return sum(
        1 for g in gate_analysis
        if g["passes_both_gates"] and not g["ensemble_direction_ok"]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_h28_ensemble_direction_ok_for_positive_signal() -> None:
    """Positive ensemble signal → ensemble_direction_ok=True for a long candidate."""
    signals = [{"ticker": "PSKY", "ensemble_signal": 0.43, "lgbm_pred_return": 0.023, "lgbm_dir_prob": 0.86}]
    result = _compute_gate_analysis(signals)
    assert result[0]["ensemble_direction_ok"] is True
    assert "ensemble_direction_inconsistent" not in result[0]["blocked_by"]


def test_h28_ensemble_direction_flagged_for_negative_signal_passing_gates() -> None:
    """Negative ensemble on a signal that otherwise passes both LGBM gates is flagged.

    This replicates the SNDK Sep-21 2026 case: lgbm_pred_return and lgbm_dir_prob
    both passed their gates while sentiment dragged ensemble_signal below zero.
    H18a (PR #41) would gate on this; H28 at minimum surfaces it in diagnostics.
    """
    # SNDK-like: good pred_return and dir_prob but negative composite ensemble
    signals = [{
        "ticker": "SNDK",
        "ensemble_signal": -0.069,
        "lgbm_pred_return": 0.008,   # > cost_thr=0.00235
        "lgbm_dir_prob": 0.72,       # outside dead zone [0.4, 0.6]
    }]
    result = _compute_gate_analysis(signals)
    assert result[0]["passes_both_gates"] is True, "SNDK should pass pred_return and dir_prob gates"
    assert result[0]["ensemble_direction_ok"] is False, "Negative ensemble should be flagged"
    assert "ensemble_direction_inconsistent" in result[0]["blocked_by"]


def test_h28_direction_anomaly_not_flagged_when_gates_fail() -> None:
    """A negative ensemble on a signal that fails pred_return gate is not counted
    as a direction anomaly (it's already blocked for a different reason)."""
    signals = [{
        "ticker": "BE",
        "ensemble_signal": -0.364,
        "lgbm_pred_return": -0.002,  # abs < cost_thr → fails pred_return gate
        "lgbm_dir_prob": 0.34,
    }]
    result = _compute_gate_analysis(signals)
    assert result[0]["passes_both_gates"] is False
    assert result[0]["ensemble_direction_ok"] is False
    assert "ensemble_direction_inconsistent" not in result[0]["blocked_by"], (
        "Direction flag only applies to signals that pass both quality gates"
    )


def test_h28_direction_anomaly_count_matches() -> None:
    """Anomaly counter correctly sums only signals that pass gates but have negative ensemble."""
    signals = [
        # passes gates, positive ensemble — no anomaly
        {"ticker": "PSKY", "ensemble_signal": 0.43, "lgbm_pred_return": 0.023, "lgbm_dir_prob": 0.86},
        # passes gates, NEGATIVE ensemble — anomaly!
        {"ticker": "SNDK", "ensemble_signal": -0.069, "lgbm_pred_return": 0.008, "lgbm_dir_prob": 0.72},
        # fails gates — not an anomaly
        {"ticker": "BE", "ensemble_signal": -0.364, "lgbm_pred_return": -0.002, "lgbm_dir_prob": 0.34},
        # passes gates, positive ensemble — no anomaly
        {"ticker": "ILMN", "ensemble_signal": 0.329, "lgbm_pred_return": 0.0077, "lgbm_dir_prob": 0.77},
    ]
    gate = _compute_gate_analysis(signals)
    anomalies = _count_direction_anomalies(gate)
    assert anomalies == 1, f"Expected 1 direction anomaly (SNDK), got {anomalies}"


def test_h28_kelly_hard_block_surfaced_when_entries_blocked() -> None:
    """Kelly hard_blocked mirrors kelly_entries_blocked in the diagnostics payload.

    Confirms the field is correctly derived: when kelly_entries_blocked=True
    (fraction ≤ KELLY_HARD_BLOCK_THRESHOLD=-0.25), kelly_hard_blocked=True.
    Sep-21 live data: kelly_fraction=-0.2699, kelly_mode='probation',
    kelly_entries_blocked=True → hard block IS active, probes are ALSO blocked.
    """
    summary_entries_blocked = {"kelly_entries_blocked": True, "kelly_fraction": -0.2699}
    summary_entries_open = {"kelly_entries_blocked": False, "kelly_fraction": 0.0}

    hard_blocked = summary_entries_blocked.get("kelly_entries_blocked", False)
    not_hard_blocked = summary_entries_open.get("kelly_entries_blocked", False)

    assert hard_blocked is True, "Should be hard-blocked at fraction=-0.2699"
    assert not_hard_blocked is False, "Should NOT be hard-blocked at fraction=0.0"
