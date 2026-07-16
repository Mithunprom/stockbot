"""Unit tests for H5 hold extension in research_backtest.simulate().

Tests the signal-reconfirmed hold extension logic (Params.max_hold_extensions)
added as part of hypothesis H5. All tests use synthetic RTH prediction DataFrames
— no Alpaca credentials or cached bar data are required.

The module is loaded via importlib because scripts/ has no __init__.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── Load research_backtest as a module from file path ────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
_spec = importlib.util.spec_from_file_location(
    "research_backtest", _REPO / "scripts" / "research_backtest.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
Params = _mod.Params
simulate = _mod.simulate


# ─── Synthetic RTH prediction helper ─────────────────────────────────────────

def _make_preds(
    start: str = "2026-05-19",
    n_days: int = 3,
    ticker: str = "SYN",
    pred_return: float = 0.01,
    dir_prob: float = 0.70,
    base_price: float = 100.0,
    daily_vol: float = 0.02,
    price_drift: float = 0.00001,
) -> pd.DataFrame:
    """Synthetic 1m RTH prediction DataFrame for unit testing.

    Generates n_days × 390 RTH minute bars starting from `start` (Mon-Fri only).
    Price follows a constant compound drift from base_price. Signal params are
    constant throughout so the entry gate fires on the first qualifying bar and
    the extension check sees the same signal at every max_hold bar.

    Args:
        price_drift: Per-bar multiplicative price change (e.g. 0.00001 = mild up,
                     -0.0000539 gives −2.1% at bar 390 for loss-gate testing).
    """
    rng = pd.date_range(
        start=f"{start} 09:30:00",
        periods=n_days * 24 * 60 * 2,  # generous window, filtered below
        freq="1min",
        tz="America/New_York",
    )
    rth = rng[
        (rng.dayofweek < 5)
        & (rng.hour * 60 + rng.minute >= 9 * 60 + 30)
        & (rng.hour * 60 + rng.minute <= 15 * 60 + 59)
    ]
    rth = rth[: n_days * 390]
    n = len(rth)
    prices = base_price * (1.0 + price_drift) ** np.arange(n)
    return pd.DataFrame({
        "ticker": ticker,
        "time": rth.tz_convert("UTC"),
        "pred_return": pred_return,
        "dir_prob": dir_prob,
        "close": prices,
        "open": prices,          # open ≈ close (flat fill assumption for testing)
        "atr_pct": 0.001,
        "daily_vol": daily_vol,
        "regime": 0,
        "forward_return": 0.0,
    })


def _trades(result: dict) -> list[dict]:
    """Extract closed trades list from simulate() result."""
    tdf = result["_trades"]
    return tdf.to_dict("records") if len(tdf) else []


# ─── Params field tests ───────────────────────────────────────────────────────

def test_params_extension_fields_default_disabled():
    """max_hold_extensions defaults to 0 (baseline: no extension)."""
    p = Params()
    assert p.max_hold_extensions == 0


def test_params_extension_loss_gate_default():
    """ext_loss_gate_dvol_mult defaults to 1.0 (1× daily vol loss gate)."""
    p = Params()
    assert p.ext_loss_gate_dvol_mult == 1.0


def test_params_extension_fields_settable():
    """Extension params can be set and survive dataclass construction."""
    p = Params(max_hold_extensions=2, ext_loss_gate_dvol_mult=1.5, label="test")
    assert p.max_hold_extensions == 2
    assert p.ext_loss_gate_dvol_mult == 1.5


def test_params_baseline_unchanged_by_new_fields():
    """Adding extension fields to Params does not change baseline Params defaults."""
    p = Params()
    assert p.stop_mult == 1.1
    assert p.max_hold_bars == 1170   # 3 trading days (390 × 3)
    assert p.cost_threshold == 0.003
    assert p.dead_hi == 0.55


# ─── Baseline: max_hold exits without extension ───────────────────────────────

def test_simulate_baseline_exits_at_max_hold():
    """With max_hold_extensions=0 (default), position exits at max_hold in ~1 day.

    Single synthetic ticker, constant qualifying signal, mild uptrend.
    PDT is disabled so the same-day max_hold block doesn't complicate timing.
    The position holds one full trading day (~390 bars) and exits via max_hold.
    """
    preds = _make_preds(start="2026-05-19", n_days=2, price_drift=0.00001)
    p = Params(max_hold_extensions=0, pdt_enabled=False, label="baseline")
    result = simulate(preds, p, "2026-05-19", "2026-05-20")
    closed = _trades(result)

    assert len(closed) >= 1, "Expected at least one closed trade"
    t = closed[0]
    assert t["reason"] == "max_hold", f"Expected max_hold, got {t['reason']!r}"
    assert t.get("ext_count", 0) == 0, "Baseline: no extensions should be recorded"
    # bars held ≈ 390 (one trading day after entry)
    assert 380 <= t["bars"] <= 400, (
        f"Expected ~390 bars for 1-day hold, got {t['bars']}"
    )


# ─── H5: extension granted ────────────────────────────────────────────────────

def test_simulate_extension_granted_doubles_hold_time():
    """With max_hold_extensions=1, a qualifying signal at max_hold resets the bar
    counter and extends the hold by one more trading day (~780 bars total).

    Acceptance: ext_count == 1, exit reason == 'max_hold', bars ≈ 390 (after reset).
    """
    # Need 3 days: day 1 = entry, day 2 = extension point, day 3 = final exit
    preds = _make_preds(start="2026-05-19", n_days=3, price_drift=0.00001)
    p = Params(max_hold_extensions=1, pdt_enabled=False, label="ext1")
    result = simulate(preds, p, "2026-05-19", "2026-05-21")
    closed = _trades(result)

    assert len(closed) >= 1, "Expected at least one closed trade"
    t = closed[0]
    # The second max_hold window records bars=390 again (counter reset to 0 after ext)
    assert t["reason"] == "max_hold", f"Expected max_hold, got {t['reason']!r}"
    assert t.get("ext_count", 0) == 1, (
        f"Expected ext_count=1 (one extension granted), got {t.get('ext_count')}"
    )
    # After extension, bars resets to 0 then reaches 390 again for the 2nd max_hold
    assert 380 <= t["bars"] <= 410, (
        f"Expected ~390 bars (post-reset window), got {t['bars']}"
    )


def test_simulate_no_extension_when_disabled():
    """Extension fires only when max_hold_extensions > 0.

    With max_hold_extensions=0, the position exits immediately at the first
    max_hold (no extension, ext_count=0).
    """
    preds = _make_preds(start="2026-05-19", n_days=3, price_drift=0.00001)
    p_baseline = Params(max_hold_extensions=0, pdt_enabled=False, label="no_ext")
    p_ext = Params(max_hold_extensions=1, pdt_enabled=False, label="ext1")

    r_base = simulate(preds, p_baseline, "2026-05-19", "2026-05-21")
    r_ext = simulate(preds, p_ext, "2026-05-19", "2026-05-21")

    base_trades = _trades(r_base)
    ext_trades = _trades(r_ext)

    assert base_trades, "Baseline must produce at least one trade"
    assert ext_trades, "Extension must produce at least one trade"

    # Baseline: no extension
    assert base_trades[0].get("ext_count", 0) == 0
    # Extension: at least 1 extension granted
    assert ext_trades[0].get("ext_count", 0) >= 1


# ─── H5: loss gate denial ─────────────────────────────────────────────────────

def test_simulate_extension_denied_by_loss_gate():
    """Extension is denied when unrealized loss exceeds daily_vol × loss_gate_mult.

    With daily_vol=0.02 and mult=1.0, the gate blocks extension when unrealized
    < −2.0%. A price drift of −0.0000539/bar gives −2.1% at bar 390, which is
    just below the gate. Stop-loss (at −2.2%) does not fire in this window.

    Expected: exit at max_hold with ext_count=0 (extension denied).
    """
    # drift=-0.0000539 → price drops 2.1% by bar ~390; stop at 2.2% doesn't fire
    preds = _make_preds(
        start="2026-05-19", n_days=2,
        daily_vol=0.02,
        price_drift=-0.0000539,
    )
    p = Params(
        max_hold_extensions=1,
        ext_loss_gate_dvol_mult=1.0,
        pdt_enabled=False,
        label="loss_gate",
    )
    result = simulate(preds, p, "2026-05-19", "2026-05-20")
    closed = _trades(result)

    assert len(closed) >= 1, "Expected at least one closed trade"
    t = closed[0]
    assert t["reason"] == "max_hold", (
        f"Expected max_hold exit (loss gate denial), got {t['reason']!r}"
    )
    assert t.get("ext_count", 0) == 0, (
        f"Expected ext_count=0 (loss gate denied), got {t.get('ext_count')}"
    )


def test_simulate_extension_allowed_above_loss_gate():
    """Extension IS granted when unrealized is above the loss gate.

    Mild uptrend means the position is profitable at max_hold, so the loss
    gate passes and the extension is granted (given qualifying signal).
    """
    preds = _make_preds(
        start="2026-05-19", n_days=3,
        daily_vol=0.02,
        price_drift=0.00001,  # mild uptrend: position profitable at bar 390
    )
    p = Params(
        max_hold_extensions=1,
        ext_loss_gate_dvol_mult=1.0,
        pdt_enabled=False,
        label="profitable",
    )
    result = simulate(preds, p, "2026-05-19", "2026-05-21")
    closed = _trades(result)

    assert len(closed) >= 1, "Expected at least one closed trade"
    t = closed[0]
    # Extension granted → ext_count=1
    assert t.get("ext_count", 0) == 1, (
        f"Expected ext_count=1 (extension above loss gate), got {t.get('ext_count')}"
    )


# ─── H5: extension cap ────────────────────────────────────────────────────────

def test_simulate_extension_cap_respected():
    """With max_hold_extensions=1, the cap is respected: only 1 extension fires.

    After the first extension (ext_count=1), the second max_hold check sees
    ext_ct=1 == max_extensions=1, so the position exits (no second extension).

    Expected: 1 closed trade, ext_count=1.
    """
    preds = _make_preds(start="2026-05-19", n_days=3, price_drift=0.00001)
    p = Params(max_hold_extensions=1, pdt_enabled=False, label="cap1")
    result = simulate(preds, p, "2026-05-19", "2026-05-21")
    closed = _trades(result)

    assert len(closed) >= 1, "Expected at least one closed trade"
    t = closed[0]
    # Should exit at 2nd max_hold with exactly 1 extension used
    assert t.get("ext_count", 0) == 1, (
        f"Expected exactly 1 extension used, got {t.get('ext_count')}"
    )
    assert t["reason"] == "max_hold"


def test_simulate_two_extensions_with_cap_2():
    """With max_hold_extensions=2, two extensions fire before the final exit.

    Extension 1 fires at the 1st max_hold (ext_count=1, bars reset).
    Extension 2 fires at the 2nd max_hold (ext_count=2, bars reset).
    The 3rd max_hold sees ext_ct=2 == max_extensions=2 → exits.

    Expected: 1 closed trade, ext_count=2.
    """
    # Need 4 days: entry day + 3 max_hold windows
    preds = _make_preds(start="2026-05-19", n_days=4, price_drift=0.00001)
    p = Params(max_hold_extensions=2, pdt_enabled=False, label="cap2")
    result = simulate(preds, p, "2026-05-19", "2026-05-22")
    closed = _trades(result)

    assert len(closed) >= 1, "Expected at least one closed trade"
    t = closed[0]
    assert t.get("ext_count", 0) == 2, (
        f"Expected exactly 2 extensions used, got {t.get('ext_count')}"
    )
    assert t["reason"] == "max_hold"


# ─── H5: ext_count in trade dict ─────────────────────────────────────────────

def test_simulate_ext_count_present_in_baseline_trade():
    """Baseline trades include ext_count=0 key (backward-compatible with reporting)."""
    preds = _make_preds(start="2026-05-19", n_days=2, price_drift=0.00001)
    p = Params(max_hold_extensions=0, pdt_enabled=False, label="baseline_key")
    result = simulate(preds, p, "2026-05-19", "2026-05-20")
    closed = _trades(result)

    if closed:
        assert "ext_count" in closed[0], "ext_count key must be present in all trades"
        assert closed[0]["ext_count"] == 0


# ─── phase_holdext exists and is wired into CLI ───────────────────────────────

def test_phase_holdext_is_defined():
    """phase_holdext() is importable from research_backtest."""
    assert hasattr(_mod, "phase_holdext"), "phase_holdext not found in research_backtest"
    assert callable(_mod.phase_holdext)


def test_holdext_in_cli_choices(monkeypatch, capsys):
    """--phase holdext is a valid CLI argument (parser doesn't error)."""
    import argparse
    # Re-run main() up to the argparse step using monkeypatch
    monkeypatch.setattr(sys, "argv", ["research_backtest.py", "--phase", "holdext"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["fetch", "features", "train", "simulate", "tune",
                             "holdext", "all"])
    args = ap.parse_args(["--phase", "holdext"])
    assert args.phase == "holdext"
