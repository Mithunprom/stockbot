"""Unit tests for the forecast module + email agent (no network)."""
from __future__ import annotations

from datetime import date

from src.analysis.forecast import (
    _clamp,
    _next_session,
    _score_direction,
    render_text,
    subject_line,
)


def test_next_session_skips_weekend():
    # Fri 2026-07-03 → next weekday is Mon 2026-07-06 (holidays ignored)
    assert _next_session(date(2026, 7, 3)) == date(2026, 7, 6)
    # Tue → Wed
    assert _next_session(date(2026, 6, 30)) == date(2026, 7, 1)


def test_clamp_bounds():
    assert _clamp(5, 0, 1) == 1
    assert _clamp(-5, 0, 1) == 0
    assert _clamp(0.5, 0, 1) == 0.5


def _tech(**over):
    base = dict(
        above_sma10=True, above_sma20=True, rsi14=64.0, chg_1d_pct=2.0,
    )
    base.update(over)
    return base


def test_score_direction_bounded_and_humble():
    ic = {"ticker_ic": 0.088, "ticker_n": 280}
    direction, prob_up, why = _score_direction(_tech(), ic, live=None)
    assert 0.45 <= prob_up <= 0.62          # never overconfident
    assert direction in {"long", "short", "neutral"}
    assert any("IC" in w for w in why)


def test_score_direction_downtrend_leans_short_or_neutral():
    ic = {"ticker_ic": -0.15, "ticker_n": 300}
    direction, prob_up, _ = _score_direction(
        _tech(above_sma10=False, above_sma20=False, rsi14=35.0), ic, live=None
    )
    assert prob_up <= 0.52                    # not bullish on a downtrend + neg IC


def test_live_signal_dominates_score():
    ic = {"ticker_ic": 0.0, "ticker_n": 0}
    up, p_up, _ = _score_direction(_tech(), ic, live={"dir_prob": 0.70})
    dn, p_dn, _ = _score_direction(_tech(), ic, live={"dir_prob": 0.30})
    assert p_up > p_dn
    assert up == "long"


def _synthetic_fc():
    return {
        "ticker": "SNDK", "generated_at": "2026-07-01T06:00:00+00:00",
        "target_session": "2026-07-01", "reference_close": 2273.73,
        "reference_bar_date": "2026-06-30", "direction": "long", "prob_up": 0.524,
        "conviction": "low", "expected_close": 2274.82, "expected_return_pct": 0.05,
        "range_1sigma": [2069.09, 2478.37], "range_1sigma_pct": 9.0,
        "rationale": ["live per-ticker IC=+0.088 (n=280)"],
        "trade_plan": {
            "side": "long", "reference_entry": 2273.73, "stop_loss_pct": 0.025,
            "trailing_stop_pct": 0.03, "take_profit_pct": 0.07,
            "stop_loss_price": 2216.89, "take_profit_price": 2432.89,
            "max_hold": "1 trading day",
        },
        "caveats": ["high vol"],
    }


def test_render_and_subject():
    fc = _synthetic_fc()
    body = render_text(fc)
    assert "SNDK" in body and "2069.09" in body and "2432.89" in body
    subj = subject_line(fc)
    assert subj.startswith("SNDK Forecast — 2026-07-01") and "low conviction" in subj


def test_agent_smtp_gating_skips_without_creds(monkeypatch):
    # Ensure no SMTP creds → agent reports not ready (never crashes)
    import src.config as cfg
    from src.agents.forecast_agent import ForecastEmailAgent

    s = cfg.get_settings()
    monkeypatch.setattr(s, "smtp_host", "", raising=False)
    monkeypatch.setattr(s, "forecast_email_to", "", raising=False)
    agent = ForecastEmailAgent(tickers=["SNDK"])
    ready, reason = agent._smtp_ready()
    assert ready is False and "SMTP" in reason
