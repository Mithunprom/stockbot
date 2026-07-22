"""Walk-forward backtest of the live system: LightGBM signal + swing-exit engine.

Reproduces the production pipeline offline with zero DB dependency:
  fetch     Alpaca IEX 1m bars (same feed production ingests) → cache
  features  compute_indicators_for_universe() — the exact production pipeline
  train     LGBMSignalModel with a hard time cutoff; predictions are only ever
            generated for bars AFTER the cutoff (out-of-sample by construction)
  simulate  event-driven replay of the deployed entry gates + swing exits
  tune      parameter grid on the first OOS half; untouched validation on the
            second half — a change ships only if it improves BOTH legs

Usage:
    python scripts/research_backtest.py --phase fetch
    python scripts/research_backtest.py --phase features
    python scripts/research_backtest.py --phase train
    python scripts/research_backtest.py --phase simulate     # baseline params
    python scripts/research_backtest.py --phase tune
    python scripts/research_backtest.py --phase all
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))  # errors only

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE = Path(os.environ.get("RESEARCH_CACHE", "/tmp/stockbot_research"))
CACHE.mkdir(parents=True, exist_ok=True)
ET = ZoneInfo("America/New_York")

# Windows are env-overridable so a retrain can be pointed at the CURRENT
# regime without editing code (RESEARCH_BARS_START=... python scripts/...).
# CACHE_TAG isolates bar/feature caches per window — reusing one cache across
# different date ranges silently trains on the wrong span.
BARS_START = os.environ.get("RESEARCH_BARS_START", "2026-03-20")   # indicator warmup
BARS_END = os.environ.get("RESEARCH_BARS_END", "2026-06-21")
TRAIN_END = os.environ.get("RESEARCH_TRAIN_END", "2026-05-08")     # train <= this date
VAL_END = os.environ.get("RESEARCH_VAL_END", "2026-05-15")         # metrics only
TUNE_END = os.environ.get("RESEARCH_TUNE_END", "2026-05-29")       # tune leg
VAL_LEG_START = os.environ.get("RESEARCH_VAL_LEG_START", "2026-06-01")
VAL_LEG_END = os.environ.get("RESEARCH_VAL_LEG_END", "2026-06-11")
FORWARD_N = 15
DIRECTION_EPSILON = 0.0001

MODEL_META = Path("models/lgbm/lgbm_ic_0.1860.json")


def universe() -> list[str]:
    with open("config/universe.json") as f:
        return [t for t in json.load(f)["symbols"] if "/" not in t]


def feature_cols() -> list[str]:
    with open(MODEL_META) as f:
        return json.load(f)["feature_cols"]


# ─── Phase: fetch ─────────────────────────────────────────────────────────────

def phase_fetch() -> None:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed

    # Env vars first (Railway worker has no .env file), .env fallback (local dev)
    import os
    keys: dict[str, str] = {
        k: os.environ[k]
        for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
        if os.environ.get(k)
    }
    if len(keys) < 2 and Path(".env").exists():
        for line in open(".env"):
            if line.startswith(("ALPACA_API_KEY", "ALPACA_SECRET_KEY")) and "=" in line:
                k, v = line.strip().split("=", 1)
                keys.setdefault(k.strip(), v.strip())
    client = StockHistoricalDataClient(keys["ALPACA_API_KEY"], keys["ALPACA_SECRET_KEY"])

    for ticker in universe():
        out = CACHE / f"bars_{ticker}.csv.gz"
        if out.exists():
            print(f"{ticker}: cached")
            continue
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=datetime.fromisoformat(BARS_START).replace(tzinfo=timezone.utc),
            end=datetime.fromisoformat(BARS_END).replace(tzinfo=timezone.utc),
            feed=DataFeed.IEX,
            adjustment=Adjustment.SPLIT,
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            print(f"{ticker}: NO DATA")
            continue
        df = bars.reset_index()
        df = df.rename(columns={"symbol": "ticker"})
        df.to_csv(out, index=False)
        print(f"{ticker}: {len(df)} bars → {out.name}")


def load_bars() -> dict[str, pd.DataFrame]:
    """RTH-filtered 1m bars per ticker, indexed by UTC time."""
    bars: dict[str, pd.DataFrame] = {}
    for ticker in universe():
        p = CACHE / f"bars_{ticker}.csv.gz"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df.index = pd.to_datetime(df.index, utc=True)
        et = df.index.tz_convert(ET)
        rth = (et.time >= dtime(9, 30)) & (et.time <= dtime(15, 59)) & (et.weekday < 5)
        df = df[rth]
        if "vwap" not in df.columns:
            df["vwap"] = df["close"]
        df["vwap"] = df["vwap"].fillna(df["close"])
        bars[ticker] = df[["open", "high", "low", "close", "volume", "vwap"]]
    return bars


def _daily_vol_map() -> dict[str, "pd.Series"]:
    """Per-ticker Series (ET date → prior-day daily ATR(14)%/close).

    Mirrors the live _compute_daily_vols daily-bar ATR. Shifted by one day so a
    bar is only ever exited using volatility known BEFORE that day — no lookahead.
    """
    out: dict[str, pd.Series] = {}
    for ticker, df in load_bars().items():
        et = df.index.tz_convert(ET)
        d = pd.DataFrame(
            {"high": df["high"].values, "low": df["low"].values,
             "close": df["close"].values},
            index=pd.DatetimeIndex(et),
        )
        daily = d.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(daily) < 15:
            continue
        prev_close = daily["close"].shift()
        tr = np.maximum(
            daily["high"] - daily["low"],
            np.maximum((daily["high"] - prev_close).abs(),
                       (daily["low"] - prev_close).abs()),
        )
        atr14 = tr.rolling(14, min_periods=5).mean()
        vol = (atr14 / daily["close"]).shift(1)   # prior day → no lookahead
        vol.index = [ts.date() for ts in daily.index]
        out[ticker] = vol
    return out


# ─── Phase: features ──────────────────────────────────────────────────────────

def phase_features() -> None:
    from src.features.indicators import compute_indicators_for_universe

    bars = load_bars()
    print(f"computing indicators for {len(bars)} tickers ...")
    results = compute_indicators_for_universe(bars, shift=True)
    cols = feature_cols()
    for ticker, feat in results.items():
        close = bars[ticker]["close"]
        fwd = close.pct_change(FORWARD_N).shift(-FORWARD_N).rename("forward_return")
        keep = [c for c in cols if c in feat.columns]
        missing = [c for c in cols if c not in feat.columns]
        out = feat[keep].copy()
        for c in missing:
            out[c] = 0.0
        out = out[cols]
        out["regime"] = feat["regime"] if "regime" in feat.columns else 1
        out["close"] = close
        out["open"] = bars[ticker]["open"]
        out["forward_return"] = fwd
        out.to_csv(CACHE / f"feat_{ticker}.csv.gz")
        if missing:
            print(f"  {ticker}: missing cols zero-filled: {missing}")
        print(f"  {ticker}: {len(out)} rows")


def load_features() -> dict[str, pd.DataFrame]:
    feats = {}
    for ticker in universe():
        p = CACHE / f"feat_{ticker}.csv.gz"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        feats[ticker] = df
    return feats


# ─── Phase: train ─────────────────────────────────────────────────────────────

def phase_train() -> None:
    from src.models.lgbm import LGBMSignalModel

    feats = load_features()
    cols = feature_cols()
    train_end = pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(days=1)
    val_end = pd.Timestamp(VAL_END, tz="UTC") + pd.Timedelta(days=1)

    tr, va = [], []
    for ticker, df in feats.items():
        d = df.dropna(subset=["forward_return"])
        tr.append(d[d.index < train_end])
        va.append(d[(d.index >= train_end) & (d.index < val_end)])
    train_df = pd.concat(tr)
    val_df = pd.concat(va)
    print(f"train rows: {len(train_df):,}   val rows: {len(val_df):,}")

    X_tr = train_df[cols].fillna(0).astype(np.float32).values
    y_tr = train_df["forward_return"].values.astype(np.float32)
    X_va = val_df[cols].fillna(0).astype(np.float32).values
    y_va = val_df["forward_return"].values.astype(np.float32)

    model = LGBMSignalModel(feature_cols=cols)
    metrics = model.train(X_tr, y_tr, X_va, y_va, DIRECTION_EPSILON)
    print(f"train_ic={metrics['train_ic']:.4f}  val_ic={metrics['val_ic']:.4f}  "
          f"val_dir_acc={metrics['val_dir_acc']:.4f}")

    # Optional deployable artifact: RESEARCH_SAVE_MODEL=models/lgbm/name
    # writes the .pkl/.json pair production loads. The research harness is
    # the only place that trains on the same feature pipeline production
    # uses, so this is the honest path from validated recipe to deployment.
    save_stem = os.environ.get("RESEARCH_SAVE_MODEL")
    if save_stem:
        stem = Path(save_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        # model.save() writes the exact dict payload LGBMSignalModel.load()
        # expects — a raw pickle.dump(model) loads as an object and breaks
        # production with KeyError('feature_cols').
        model.save(Path(str(stem) + ".pkl"))
        meta = {
            "feature_cols": cols,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_end": TRAIN_END, "val_end": VAL_END,
            "bars_start": BARS_START, "bars_end": BARS_END,
            "train_rows": int(len(train_df)), "val_rows": int(len(val_df)),
            **{k: float(v) for k, v in metrics.items()},
        }
        with open(Path(str(stem) + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"saved model → {stem}.pkl")

    # OOS predictions: strictly AFTER the validation window
    dvol_map = _daily_vol_map()
    rows = []
    for ticker, df in feats.items():
        oos = df[df.index >= val_end]
        if oos.empty:
            continue
        X = oos[cols].fillna(0).astype(np.float32).values
        pred = model.regressor.predict(X)
        prob = model.classifier.predict_proba(X)[:, 1]
        # True daily vol per row (map by prior-day ET date); fall back to the
        # 1m ATR × sqrt(390) proxy where a daily value isn't available.
        atr1m = oos["atr_pct"].values.astype(float)
        proxy = np.clip(atr1m, 0.0002, 0.01) * DVOL
        dvser = dvol_map.get(ticker)
        if dvser is not None:
            oos_dates = oos.index.tz_convert(ET).date
            daily_vol = np.array([dvser.get(d, np.nan) for d in oos_dates], dtype=float)
            daily_vol = np.where(np.isfinite(daily_vol) & (daily_vol > 0), daily_vol, proxy)
        else:
            daily_vol = proxy
        rows.append(pd.DataFrame({
            "ticker": ticker,
            "time": oos.index,
            "pred_return": pred,
            "dir_prob": prob,
            "close": oos["close"].values,
            "open": oos["open"].values,
            "atr_pct": oos["atr_pct"].values,
            "daily_vol": daily_vol,
            "regime": oos["regime"].values,
            "forward_return": oos["forward_return"].values,
        }))
    if not rows:
        # Training ran right up to the end of available data, so there is no
        # out-of-sample span left to predict. Benign for a final production
        # retrain (the recipe is validated separately with an earlier cutoff).
        print("no OOS window after val_end — skipping prediction export")
        return
    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(CACHE / "preds_oos.csv.gz", index=False)

    # quick OOS IC sanity
    from scipy import stats
    d = preds.dropna(subset=["forward_return"])
    ic = stats.spearmanr(d.pred_return, d.forward_return).statistic
    dacc = ((d.pred_return > 0) == (d.forward_return > 0)).mean()
    print(f"OOS rows: {len(preds):,}  OOS IC15={ic:.4f}  dir_acc={dacc:.4f}")
    json.dump({"oos_ic15": float(ic), "oos_dir_acc": float(dacc), **metrics},
              open(CACHE / "train_metrics.json", "w"), indent=1)


# ─── Phase: simulate ──────────────────────────────────────────────────────────

@dataclass
class Params:
    stop_mult: float = 1.1
    trail_mult: float = 1.2
    tp_mult: float = 3.0
    sl_floor: float = 0.010
    sl_cap: float = 0.100
    ts_floor: float = 0.008
    ts_cap: float = 0.100
    tp_floor: float = 0.020
    tp_cap: float = 0.200
    max_hold_bars: int = 1170
    stag_bars: int = 780
    stag_pnl: float = 0.004
    reversal_bars: int = 45
    catastrophic_mult: float = 2.0
    cost_threshold: float = 0.003
    dead_hi: float = 0.55
    dead_lo: float = 0.40
    entry_start: tuple = (9, 40)
    entry_end: tuple = (15, 30)
    max_entries_tick: int = 2
    max_pos: int = 4
    heat_ceiling: float = 0.60
    sector_cap_n: int = 2
    cooldown: int = 60
    trades_per_day: int = 4
    pdt_max: int = 3
    pdt_enabled: bool = True      # False = equity ≥ $25k (no day-trade limits)
    dyn_thresh_pct: float | None = None  # e.g. 92 → cost threshold = trailing
                                  # 92nd percentile of |pred| (self-calibrating)
    cost_bps: float = 2.0
    label: str = "baseline"


DVOL = 19.75
REGIME_SCALE = {0: 1.00, 1: 0.70, 2: 0.50}


def _exits(daily_vol: float, p: Params) -> tuple[float, float, float]:
    # Mirrors live signal_loop._atr_exits: consumes a TRUE daily-vol fraction.
    dv = min(max(daily_vol, DAILY_VOL_FLOOR), DAILY_VOL_CEIL)
    sl = min(max(dv * p.stop_mult, p.sl_floor), p.sl_cap)
    ts = min(max(dv * p.trail_mult, p.ts_floor), p.ts_cap)
    tp = min(max(dv * p.tp_mult, p.tp_floor), p.tp_cap)
    return sl, ts, tp


# Live exit path uses true daily ATR (daily bars). The backtest derives the same
# from the cached 1m bars resampled to daily (see _daily_vol_map). The 1m ATR ×
# sqrt(390) proxy underestimates gappy names, so the exits are computed on the
# real daily figure — with the 1m proxy as a per-row fallback.
DAILY_VOL_FLOOR = 0.005
DAILY_VOL_CEIL = 0.15


def _is_confirmed_reversal_long(
    pred_return: float,
    dir_prob: float,
    threshold: float,
    dead_lo: float,
) -> bool:
    """True when this bar is a CONFIRMED bearish signal against a long position.

    Mirrors production signal_loop._confirmed_opposite_signal:
    opposite direction + magnitude above the cost threshold + dir_prob below
    the lower dead-zone bound (strong bearish conviction, not noise).

    Pre-v0.3.4 used a bare sign check (pred_return < 0) that misfired on
    every momentary sign flip — the root cause of the 25-minute median hold
    diagnosed on 2026-07-07. The production fix landed in v0.3.4; this
    backtest was not updated until now.
    """
    if pred_return >= 0:
        return False
    if abs(pred_return) <= threshold:
        return False
    return dir_prob < dead_lo


def simulate(preds: pd.DataFrame, p: Params, start: str, end: str,
             capital: float = 10_000.0) -> dict:
    from src.execution.position_sizer import SmartPositionSizer, SECTOR_MAP

    sizer = SmartPositionSizer(mode="paper")

    # Self-calibrating threshold: trailing percentile of |pred| over the prior
    # 3 trading days (computed on the FULL prediction history so leg-start days
    # have context). Robust to model-retrain magnitude shifts.
    day_thr: dict = {}
    if p.dyn_thresh_pct:
        full = preds[["time", "pred_return"]].copy()
        full["date_et"] = full.time.dt.tz_convert(ET).dt.date
        dates = sorted(full.date_et.unique())
        for i, d in enumerate(dates):
            prior = full[full.date_et.isin(dates[max(0, i - 3):i])]
            if len(prior) > 100:
                base = float(np.nanpercentile(prior.pred_return.abs(), p.dyn_thresh_pct))
                day_thr[d] = max(base, 0.002)

    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = preds[(preds.time >= lo) & (preds.time < hi)].copy()
    df = df.sort_values("time")

    # next bar open per ticker (fill price for orders placed on this bar)
    df["next_open"] = df.groupby("ticker")["open"].shift(-1)
    df["next_open"] = df["next_open"].fillna(df["close"])

    by_ts: dict = {}
    for r in df.itertuples(index=False):
        by_ts.setdefault(r.time, []).append(r)
    timestamps = sorted(by_ts)

    cash = capital
    positions: dict[str, dict] = {}
    cooldowns: dict[str, int] = {}
    trades: list[dict] = []
    equity_by_day: dict[date, float] = {}
    minute_equity: list[float] = []
    daytrade_dates: list[date] = []
    trades_today = 0
    cur_day: date | None = None
    cost = p.cost_bps / 10_000.0

    def day_trades_used(today: date) -> int:
        cutoff = np.busday_offset(today, -5, roll="backward").astype("datetime64[D]")
        return sum(1 for d in daytrade_dates if np.datetime64(d) > cutoff)

    for ts in timestamps:
        rows = by_ts[ts]
        ts_et = ts.tz_convert(ET)
        today = ts_et.date()
        if today != cur_day:
            cur_day = today
            trades_today = 0
        for t in list(cooldowns):
            cooldowns[t] -= 1
            if cooldowns[t] <= 0:
                del cooldowns[t]

        price_now = {r.ticker: r.close for r in rows}
        fill_next = {r.ticker: r.next_open for r in rows}
        row_by_ticker = {r.ticker: r for r in rows}

        # mark to market
        pv = cash + sum(pos["qty"] * price_now.get(t, pos["last"])
                        for t, pos in positions.items())
        for t, pos in positions.items():
            if t in price_now:
                pos["last"] = price_now[t]
        minute_equity.append(pv)
        equity_by_day[today] = pv

        # ── exits ──
        for t in list(positions):
            if t not in price_now:
                continue
            pos = positions[t]
            px = price_now[t]
            pos["bars"] += 1
            pos["peak"] = max(pos["peak"], px)
            r = row_by_ticker[t]
            sl, tsl, tp = _exits(r.daily_vol if np.isfinite(r.daily_vol) else 0.03, p)
            unreal = (px - pos["entry_px"]) / pos["entry_px"]
            reason = None
            if unreal < -sl:
                reason = "stop_loss"
            elif unreal > tp:
                reason = "take_profit"
            elif (pos["peak"] - px) / pos["peak"] > tsl:
                reason = "trailing_stop"
            elif pos["bars"] >= p.max_hold_bars:
                reason = "max_hold"
            elif pos["bars"] >= p.stag_bars and abs(unreal) < p.stag_pnl:
                reason = "stagnation"
            else:
                # Confirmed reversal: opposite signal must clear quality gates
                # (mirrors production _confirmed_opposite_signal — sign-only
                # check was the pre-v0.3.4 bug that turned swing trades into
                # 25-minute scalps).
                rev_thr = day_thr.get(today, p.cost_threshold) if p.dyn_thresh_pct else p.cost_threshold
                if _is_confirmed_reversal_long(r.pred_return, r.dir_prob, rev_thr, p.dead_lo):
                    pos["rev"] += 1
                else:
                    pos["rev"] = 0
                if pos["rev"] >= p.reversal_bars:
                    reason = "signal_reversal"
            if reason is None:
                continue
            # PDT: same-day exits only for stops (budget≥1) / TP (budget≥2)
            if p.pdt_enabled and pos["entry_date"] == today:
                budget = p.pdt_max - day_trades_used(today)
                catastrophic = unreal < -(sl * p.catastrophic_mult)
                if catastrophic:
                    reason = "stop_loss"
                allowed = (reason == "stop_loss" and budget >= 1) or \
                          (reason == "take_profit" and budget >= 2)
                if not allowed:
                    continue
                daytrade_dates.append(today)
            fpx = fill_next[t] * (1 - cost)
            pnl = pos["qty"] * (fpx - pos["entry_px"])
            cash += pos["qty"] * fpx
            trades.append({
                "ticker": t, "entry": pos["entry_ts"], "exit": ts,
                "entry_px": pos["entry_px"], "exit_px": fpx,
                "pnl": pnl, "pnl_pct": fpx / pos["entry_px"] - 1,
                "notional": pos["qty"] * pos["entry_px"],
                "bars": pos["bars"], "reason": reason,
            })
            del positions[t]
            cooldowns[t] = p.cooldown

        # ── entries ──
        in_window = (dtime(*p.entry_start) <= ts_et.time() <= dtime(*p.entry_end))
        if not in_window:
            continue
        pv = cash + sum(pos["qty"] * price_now.get(t, pos["last"])
                        for t, pos in positions.items())
        thr = day_thr.get(today, p.cost_threshold) if p.dyn_thresh_pct else p.cost_threshold
        cands = [r for r in rows
                 if r.ticker not in positions
                 and r.ticker not in cooldowns
                 and np.isfinite(r.pred_return)
                 and r.pred_return > thr
                 and r.dir_prob > p.dead_hi]
        cands.sort(key=lambda r: abs(r.pred_return), reverse=True)
        entered = 0
        for r in cands:
            if entered >= p.max_entries_tick or trades_today >= p.trades_per_day:
                break
            if len(positions) >= p.max_pos:
                break
            heat = sum(pos["qty"] * price_now.get(t, pos["last"])
                       for t, pos in positions.items()) / max(pv, 1.0)
            if heat >= p.heat_ceiling:
                break
            sector = SECTOR_MAP.get(r.ticker, "other")
            n_sector = sum(1 for t in positions
                           if SECTOR_MAP.get(t, "other") == sector)
            if n_sector >= p.sector_cap_n:
                continue
            sector_notionals: dict[str, float] = {}
            for t, pos in positions.items():
                s = SECTOR_MAP.get(t, "other")
                sector_notionals[s] = sector_notionals.get(s, 0.0) + \
                    pos["qty"] * price_now.get(t, pos["last"])
            sizing = sizer.compute(
                ticker=r.ticker, dir_prob=float(r.dir_prob),
                pred_return=float(r.pred_return),
                atr_pct=float(r.atr_pct) if np.isfinite(r.atr_pct) else 0.001,
                price=float(r.close), portfolio_value=pv,
                portfolio_heat=heat, sector_notionals=sector_notionals,
                kelly_fraction=0.0,
            )
            if sizing is None:
                continue
            notional = sizing.notional * REGIME_SCALE.get(int(r.regime), 0.7)
            if notional < 500.0 or notional > cash:
                continue
            fpx = fill_next[r.ticker] * (1 + cost)
            qty = notional / fpx
            cash -= qty * fpx
            positions[r.ticker] = {
                "qty": qty, "entry_px": fpx, "entry_ts": ts,
                "entry_date": today, "peak": fpx, "bars": 0, "rev": 0,
                "last": fpx,
            }
            entered += 1
            trades_today += 1

    # liquidate leftovers at last price (mark, not a trade)
    final_pv = cash + sum(pos["qty"] * pos["last"] for pos in positions.values())

    daily = pd.Series(equity_by_day).sort_index()
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else 0.0
    eq = pd.Series(minute_equity)
    dd = float(((eq.cummax() - eq) / eq.cummax()).max()) if len(eq) else 0.0
    tdf = pd.DataFrame(trades)
    if not len(tdf) or "pnl" not in tdf.columns:
        # No trades in this window (e.g. it precedes the train cutoff, so no
        # out-of-sample predictions exist). Report zeros — never crash the run.
        return {
            "label": p.label, "start": start, "end": end,
            "total_return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "n_trades": 0,
            "open_at_end": len(positions), "avg_notional_pct": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "avg_hold_bars": 0,
            "exit_reasons": {}, "note": "no trades in window",
            "_trades": tdf,
        }
    wins = tdf[tdf.pnl > 0].pnl.sum()
    losses = -tdf[tdf.pnl < 0].pnl.sum()
    return {
        "label": p.label, "start": start, "end": end,
        "total_return_pct": round((final_pv / capital - 1) * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(dd * 100, 2),
        "profit_factor": round(wins / losses, 2) if losses > 0 else float("inf"),
        "win_rate": round(float((tdf.pnl > 0).mean()), 3),
        "n_trades": len(tdf),
        "open_at_end": len(positions),
        # The empty-tdf early return above makes per-field len() guards
        # redundant (PR #13 and main fixed this crash independently).
        "avg_notional_pct": round(float(tdf.notional.mean()) / capital * 100, 1),
        "avg_win_pct": round(float(tdf[tdf.pnl > 0].pnl_pct.mean()) * 100, 2) if (tdf.pnl > 0).any() else 0.0,
        "avg_loss_pct": round(float(tdf[tdf.pnl < 0].pnl_pct.mean()) * 100, 2) if (tdf.pnl < 0).any() else 0.0,
        "avg_hold_bars": round(float(tdf.bars.mean()), 0),
        "exit_reasons": tdf.reason.value_counts().to_dict(),
        "_trades": tdf,
    }


def load_preds() -> pd.DataFrame:
    df = pd.read_csv(CACHE / "preds_oos.csv.gz", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def phase_simulate() -> None:
    preds = load_preds()
    for (s, e, tag) in [
        (os.environ.get("RESEARCH_TUNE_LEG_START", "2026-05-18"), TUNE_END, "tune-leg"),
        (VAL_LEG_START, VAL_LEG_END, "val-leg"),
    ]:
        res = simulate(preds, Params(label=f"baseline {tag}"), s, e)
        res.pop("_trades")
        print(json.dumps(res, indent=1))


def phase_tune() -> None:
    preds = load_preds()
    grid = []
    for stop, tp, trail, hold_d, thr, dh in itertools.product(
        [0.8, 1.1, 1.5], [1.6, 2.2, 3.3], [0.6, 0.8, 1.2],
        [1, 3, 5], [0.002, 0.003, 0.005], [0.55, 0.60],
    ):
        grid.append(Params(
            stop_mult=stop, tp_mult=tp, trail_mult=trail,
            max_hold_bars=hold_d * 390,
            stag_bars=min(780, hold_d * 390),
            cost_threshold=thr, dead_hi=dh,
            label=f"s{stop}_t{tp}_tr{trail}_h{hold_d}_c{thr}_d{dh}",
        ))
    print(f"grid: {len(grid)} combos")
    rows = []
    for i, p in enumerate(grid):
        r = simulate(preds, p, "2026-05-18", TUNE_END)
        r.pop("_trades")
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(grid)}")
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    out.to_csv(CACHE / "tune_results.csv", index=False)
    print(out.head(15).to_string())

    # top-5 by tune sharpe (with ≥8 trades) → validation leg
    top = out[out.n_trades >= 8].head(5)
    print("\n=== validation leg (untouched) ===")
    val_rows = []
    for lbl in top.label:
        p = next(g for g in grid if g.label == lbl)
        r = simulate(preds, replace(p, label=lbl + "|VAL"), VAL_LEG_START, VAL_LEG_END)
        r.pop("_trades")
        val_rows.append(r)
    base_t = simulate(preds, Params(label="baseline|TUNE"),
                      os.environ.get("RESEARCH_TUNE_LEG_START", "2026-05-18"), TUNE_END)
    base_v = simulate(preds, Params(label="baseline|VAL"), VAL_LEG_START, VAL_LEG_END)
    base_t.pop("_trades"); base_v.pop("_trades")
    val_rows.extend([base_t, base_v])
    vout = pd.DataFrame(val_rows)
    vout.to_csv(CACHE / "validation_results.csv", index=False)
    print(vout.to_string())

    # Machine-readable summary for agent_worker.extract_metrics — the tables
    # above are for humans; the worker's regexes need key=value text. Report
    # the VALIDATION-leg baseline (the honest out-of-sample number), never
    # the tuned-leg figures.
    print(
        f"\nSUMMARY (baseline validation leg {base_v['start']}→{base_v['end']}): "
        f"sharpe={base_v['sharpe']} "
        f"pf={base_v['profit_factor']} "
        f"return={base_v['total_return_pct']}% "
        f"drawdown={base_v['max_dd_pct']}% "
        f"n_trades={base_v['n_trades']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["fetch", "features", "train", "simulate", "tune", "all"])
    args = ap.parse_args()
    phases = {
        "fetch": phase_fetch, "features": phase_features, "train": phase_train,
        "simulate": phase_simulate, "tune": phase_tune,
    }
    if args.phase == "all":
        for fn in phases.values():
            fn()
    else:
        phases[args.phase]()


if __name__ == "__main__":
    main()
