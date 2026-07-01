"""Next-session single-ticker forecast — shared by the CLI script and the
daily ForecastEmailAgent so there is exactly ONE implementation.

Combines three grounded inputs (see build_forecast):
  1. Model edge   — live per-ticker IC from the running bot (/admin/ic/report)
  2. Live signal  — fresh LightGBM pred_return/dir_prob if available (mkt hours)
  3. Technicals   — recent daily bars (trend, RSI, realised vol, ATR) via yfinance

Exit geometry mirrors src/agents/signal_loop.py _atr_exits so the trade plan in
the forecast matches what the live system would actually do. Fails soft: any
missing input degrades confidence rather than raising.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BOT_URL = "https://stockbot-production-cbde.up.railway.app"

# Bot exit geometry (mirrors signal_loop.py _atr_exits, 2026-06-30).
DAILY_VOL_SQRT_BARS = 19.75
SL_DVOL, TS_DVOL, TP_DVOL = 1.1, 1.2, 3.0
SL_FLOOR, SL_CAP = 0.010, 0.025
TS_FLOOR, TS_CAP = 0.008, 0.030
TP_FLOOR, TP_CAP = 0.020, 0.070

FORECAST_DIR = Path("reports/forecasts")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _next_session(d: date) -> date:
    """Next weekday (ignores exchange holidays)."""
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:
        nd += timedelta(days=1)
    return nd


def _fetch_live_ic(ticker: str) -> dict:
    out = {"ticker_ic": None, "ticker_n": None, "ticker_dir_acc": None,
           "rolling_7d_ic": None, "rolling_30d_ic": None, "source": "bot"}
    try:
        with urllib.request.urlopen(f"{BOT_URL}/admin/ic/report", timeout=20) as r:
            d = json.load(r)
        bt = (d.get("by_ticker") or {}).get(ticker) or {}
        out.update(ticker_ic=bt.get("ic"), ticker_n=bt.get("n"),
                   ticker_dir_acc=bt.get("dir_acc"),
                   rolling_7d_ic=d.get("rolling_7d_ic"),
                   rolling_30d_ic=d.get("rolling_30d_ic"))
    except Exception as exc:
        out["source"] = f"unavailable ({exc})"
    return out


def _fetch_live_signal(ticker: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{BOT_URL}/diagnostics", timeout=20) as r:
            d = json.load(r)
        for g in (d.get("pipeline_a", {}).get("signal_gate_analysis") or []):
            if g.get("ticker") == ticker:
                return {"pred_return": g.get("lgbm_pred_return"),
                        "dir_prob": g.get("lgbm_dir_prob"),
                        "would_trade": g.get("would_trade")}
    except Exception:
        pass
    return None


def _technicals(ticker: str) -> dict:
    import yfinance as yf
    df = yf.download(ticker, period="3mo", interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty or len(df) < 25:
        raise RuntimeError(f"insufficient price history for {ticker}")
    c = df["Close"].squeeze().astype(float)
    h = df["High"].squeeze().astype(float)
    l = df["Low"].squeeze().astype(float)
    last = float(c.iloc[-1])
    sma10, sma20 = float(c.tail(10).mean()), float(c.tail(20).mean())
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(),
                                      (l - c.shift()).abs())).dropna()
    atr14 = float(tr.tail(14).mean())
    daily_vol = float(c.pct_change().dropna().tail(20).std())
    dlt = c.diff()
    up = float(dlt.clip(lower=0).tail(14).mean())
    dn = float((-dlt.clip(upper=0)).tail(14).mean())
    rsi = 100 - 100 / (1 + up / dn) if dn else 100.0
    return {
        "as_of_bar": str(df.index[-1].date()),
        "last_close": round(last, 2),
        "chg_1d_pct": round((last / float(c.iloc[-2]) - 1) * 100, 2),
        "chg_5d_pct": round((last / float(c.iloc[-6]) - 1) * 100, 2),
        "chg_20d_pct": round((last / float(c.iloc[-21]) - 1) * 100, 2),
        "sma10": round(sma10, 2), "sma20": round(sma20, 2),
        "above_sma10": last > sma10, "above_sma20": last > sma20,
        "rsi14": round(rsi, 1),
        "atr14_pct": round(atr14 / last * 100, 2),
        "daily_vol_pct": round(daily_vol * 100, 2),
    }


def _score_direction(tech, ic, live) -> tuple[str, float, list[str]]:
    """Composite directional lean → (direction, prob_up, rationale).

    Deliberately humble: with ~52% realised dir-accuracy the output is a small
    tilt, not a confident call. prob_up is bounded to [0.45, 0.62].
    """
    score, why = 0.0, []
    if live and live.get("dir_prob") is not None:
        dp = float(live["dir_prob"])
        score += (dp - 0.5) * 4
        why.append(f"live LightGBM dir_prob={dp:.2f}")
    tic = ic.get("ticker_ic")
    if tic is not None:
        score += float(tic) * 2
        why.append(f"live per-ticker IC={float(tic):+.3f} (n={ic.get('ticker_n')})")
    if tech["above_sma10"] and tech["above_sma20"]:
        score += 0.5; why.append("uptrend (above SMA10 & SMA20)")
    elif not tech["above_sma10"] and not tech["above_sma20"]:
        score -= 0.5; why.append("downtrend (below SMA10 & SMA20)")
    if tech["rsi14"] >= 70:
        score -= 0.3; why.append(f"overbought RSI={tech['rsi14']}")
    elif tech["rsi14"] <= 30:
        score += 0.3; why.append(f"oversold RSI={tech['rsi14']}")
    if tech["chg_1d_pct"] >= 8:
        score -= 0.2; why.append(f"prior +{tech['chg_1d_pct']}% pop → give-back risk")
    prob_up = _clamp(0.5 + score * 0.05, 0.45, 0.62)
    direction = "long" if prob_up >= 0.52 else ("short" if prob_up <= 0.48 else "neutral")
    return direction, round(prob_up, 3), why


def build_forecast(ticker: str) -> dict:
    """Build a structured, checkable next-session forecast for `ticker`."""
    tech = _technicals(ticker)
    ic = _fetch_live_ic(ticker)
    live = _fetch_live_signal(ticker)
    direction, prob_up, why = _score_direction(tech, ic, live)

    ref = tech["last_close"]
    sigma = tech["daily_vol_pct"] / 100
    drift = (prob_up - 0.5) * 0.02
    exp_close = round(ref * (1 + drift), 2)

    dvol = _clamp(sigma / DAILY_VOL_SQRT_BARS, 0.0002, 0.01) * DAILY_VOL_SQRT_BARS
    sl = _clamp(dvol * SL_DVOL, SL_FLOOR, SL_CAP)
    ts = _clamp(dvol * TS_DVOL, TS_FLOOR, TS_CAP)
    tp = _clamp(dvol * TP_DVOL, TP_FLOOR, TP_CAP)
    long_ = direction != "short"
    plan = {
        "side": direction,
        "reference_entry": ref,
        "stop_loss_pct": round(sl, 4),
        "trailing_stop_pct": round(ts, 4),
        "take_profit_pct": round(tp, 4),
        "stop_loss_price": round(ref * (1 - sl if long_ else 1 + sl), 2),
        "take_profit_price": round(ref * (1 + tp if long_ else 1 - tp), 2),
        "max_hold": "1 trading day",
    }

    target = _next_session(date.fromisoformat(tech["as_of_bar"]))
    return {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_session": str(target),
        "reference_close": ref,
        "reference_bar_date": tech["as_of_bar"],
        "direction": direction,
        "prob_up": prob_up,
        "conviction": ("low" if abs(prob_up - 0.5) < 0.04 else "moderate"),
        "expected_close": exp_close,
        "expected_return_pct": round(drift * 100, 2),
        "range_1sigma": [round(ref * (1 - sigma), 2), round(ref * (1 + sigma), 2)],
        "range_1sigma_pct": round(sigma * 100, 2),
        "rationale": why,
        "trade_plan": plan,
        "inputs": {"technicals": tech, "model_ic": ic, "live_signal": live},
        "caveats": [
            f"{ticker} realised daily vol ~{tech['daily_vol_pct']}% >> the bot's "
            f"{plan['stop_loss_pct']*100:.1f}% stop cap → high stop-out / gap risk.",
            "Model dir-accuracy on this name is ~52%; treat direction as a weak tilt.",
        ],
        "scoring_rule": {
            "direction_hit": "realised next-session close vs reference_close matches `direction`",
            "in_range": "realised close within range_1sigma",
        },
    }


def render_text(fc: dict) -> str:
    """Plain-text render (email body / stdout)."""
    p = fc["trade_plan"]
    arrow = {"long": "▲ UP", "short": "▼ DOWN", "neutral": "► FLAT"}[fc["direction"]]
    lines = [
        f"{fc['ticker']} next-session forecast — target {fc['target_session']}",
        f"(generated {fc['generated_at'][:16]}Z, ref close {fc['reference_close']} "
        f"on {fc['reference_bar_date']})",
        "",
        f"Direction : {arrow}   |  P(up)={fc['prob_up']}  conviction={fc['conviction']}",
        f"Expected  : close ~{fc['expected_close']} ({fc['expected_return_pct']:+}%)",
        f"1σ range  : {fc['range_1sigma'][0]} – {fc['range_1sigma'][1]}  (±{fc['range_1sigma_pct']}%)",
        "",
        "Trade plan (if the bot enters, per live rules):",
        f"  side {p['side']} | stop {p['stop_loss_pct']*100:.1f}% ({p['stop_loss_price']}) "
        f"| TP {p['take_profit_pct']*100:.1f}% ({p['take_profit_price']}) "
        f"| trail {p['trailing_stop_pct']*100:.1f}% | max hold {p['max_hold']}",
        "",
        "Why: " + "; ".join(fc["rationale"]),
        "",
        "Caveats:",
    ] + [f"  - {c}" for c in fc["caveats"]] + [
        "",
        "Scoring: direction hit = next close vs "
        f"{fc['reference_close']} matches {fc['direction']}; "
        f"in-range = close within {fc['range_1sigma'][0]}–{fc['range_1sigma'][1]}.",
        "—", "Automated forecast from StockBot. Not investment advice.",
    ]
    return "\n".join(lines)


def subject_line(fc: dict) -> str:
    arrow = {"long": "▲ UP", "short": "▼ DOWN", "neutral": "► FLAT"}[fc["direction"]]
    return f"{fc['ticker']} Forecast — {fc['target_session']} ({arrow}, {fc['conviction']} conviction)"


def save_forecast(fc: dict, out_dir: Path = FORECAST_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fc['ticker']}_{fc['target_session']}.json"
    out_path.write_text(json.dumps(fc, indent=2))
    return out_path
