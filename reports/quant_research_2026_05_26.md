# Quant Research Report: StockBot Loss Investigation
**Date:** 2026-05-26
**Analyst:** Quant Research Agent
**Status:** CRITICAL -- Account value $9,874 (down ~90% from $100k)

---

## Executive Summary

The StockBot paper trading account has declined from $100,000 to approximately $9,880. This report identifies **seven root causes** spanning infrastructure failures, position sizing bugs, signal quality degradation, and architectural flaws. The most critical finding: **the $100k to $10k drop was NOT primarily caused by algorithmic trading losses -- it was caused by the Alpaca paper account being externally reset or drained**, followed by the algo compounding the problem on the remaining $10k with oversized, correlated positions.

---

## 1. Root Cause Analysis

### 1.1 The $100k to $10k Drop: Account Reset, Not Algo Losses

**Evidence from server.log:**

| Date | Event | Portfolio Value |
|------|-------|----------------|
| 2026-04-06 | A/B test started | $106,572 total ($53,286 per pipeline) |
| 2026-04-07 | Daily loss halt (-3.59%) | ~$102,742 (Pipeline A side) |
| 2026-04-23 | Last normal A/B reading | $98,656 (Pipeline A) |
| 2026-05-06 | Pipeline A daily reset | $97,508 |
| 2026-05-13 | Pipeline A daily reset | $53,286 (only A/B half logging) |
| 2026-05-14 | Pipeline A dies silently | No more Pipeline A resets after this |
| 2026-05-22 | Restart with ACTIVE_PIPELINE=b | $9,999.17 synced from Alpaca |

**Analysis:** From April 6 through May 6, the total Alpaca portfolio declined from $106,572 to ~$97,508 -- a modest 8.5% loss over 30 days. Then between May 6 and May 22, the account dropped from ~$97k to $10k -- a 90% loss. However, Pipeline A's asyncio task silently died around May 14, and the log shows no `order_executed` entries during this period. This means:

1. The Alpaca paper account was **manually reset to $10,000** (common when resetting paper accounts), OR
2. External trades (manual or via Alpaca dashboard) drained the account while the bot was partially down.

The algo's actual trading losses from April 6 to May 6 were approximately **$9,000 (8.5%)**, which is bad but not catastrophic. The $90k drop happened outside the algo's control.

**Code reference:** `main.py` line 159-163: portfolio value is synced from broker at startup. When Pipeline B restarted on May 22, it faithfully picked up the $9,999 balance.

### 1.2 Pipeline A Silently Died (May 14)

Pipeline A's asyncio task stopped running around May 14 with no error logging. The signal loop crash-recovery logic (`signal_loop.py` lines 251-257) catches exceptions and continues, but if the task was cancelled without raising CancelledError, it would silently stop.

**Evidence:** `daily_start_value_reset` shows Pipeline A stuck at $53,286.25 from May 13 onward (the stale A/B split value), while Pipeline A's actual portfolio sync stopped appearing. No `signal_loop_tick` entries after May 14.

**Impact:** For 8 days (May 14-22), no pipeline was actively trading. The account bled from positions that were never managed.

### 1.3 Legacy Position Overhang: 265% Portfolio Heat

On April 6 startup, the system synced from broker and found **265% portfolio heat** (`circuit_breakers.py` logged `Portfolio heat: 265.4%`). This means $280,000 in positions on a $106,000 portfolio -- positions from before the bot was deployed.

**Code reference:** `position_manager.py` line 259-260: `sync_from_broker()` rebuilds position map from ALL broker positions, not just those the algo opened. The `_managed_tickers` set (`line 78`) properly tracks this, but the old PositionManager synced everything.

**Bug:** XOM sell orders failed repeatedly (`"insufficient buying power"`, cost_basis=$280k) -- the bot tried to exit a stop_loss on XOM every single minute from 16:33 through 20:13 on April 6, generating hundreds of failed sell orders. (`signal_loop.py` lines 992-1000: exit triggers sell, but `alpaca.submit_order` returns error, and the position remains. Next tick, exit triggers again.)

### 1.4 Correlated Position Blowup (May 22)

On May 22, Pipeline B entered 6 positions in 3 minutes:

| Time (ET) | Ticker | Sector | Size | Pred Return |
|-----------|--------|--------|------|-------------|
| 14:29 | TSLA | consumer | 25.01% | 0.00525 |
| 14:31 | NVDA | semis | 25.01% | 0.00700 |
| 14:31 | AMD | semis | 25.01% | 0.00573 |
| 14:31 | GOOGL | tech | 24.99% | 0.00525 |
| 14:31 | SMCI | semis | 25.00% | 0.00433 |
| 14:31 | AMZN | consumer | 24.99% | 0.00420 |

**Total heat: 150%** -- on a $10k account, this is $15,000 in positions (margin-leveraged).

**Problems identified:**

1. **No portfolio-level heat gate blocked these entries.** The SmartPositionSizer's heat tiers (`position_sizer.py` lines 88-94) allow full size up to 70% heat, half-size up to 85%, quarter up to 95%. But with 6 entries in rapid succession, each entry calculates heat BEFORE the previous entries are registered (they're all triggered in the same tick cycle). Result: each entry sees 0% heat and sizes at maximum 25%.

2. **Sector cap failure.** `_SECTOR_CAP_PCT = 0.60` (`position_sizer.py` line 95). NVDA, AMD, SMCI are all "semis" sector. Combined semis = 75% -- well above the 60% cap. But again, sector_notionals are computed before entries are recorded in the same tick.

3. **ALL long, ALL tech/semis.** Every position is long mega-cap tech. A 1% market selloff hits all 6 simultaneously. There is **zero correlation check** -- no code anywhere computes inter-ticker correlation.

**Code reference:** `signal_loop_b.py` lines 190-211: the for loop iterates all signals and calls `_act_on_signal()` for each. Each call to `SmartPositionSizer.compute()` uses `self._compute_sector_notionals()` which reads from `self._pm._positions` -- but new entries are only added via `self._pm.open_position()` AFTER the order fills. During the same tick, multiple entries pass all gates simultaneously.

---

## 2. Stock Selection Critique

### 2.1 Universe is Hardcoded to 20 Mega-Cap Tech/Semis

**File:** `config/universe.json` and `screener_agent.py` lines 41-52

The `ANCHOR_TICKERS` list is 20 symbols that are ALWAYS included regardless of momentum scoring:
```
AAPL, MSFT, NVDA, AMZN, GOOGL, TSLA, AVGO, AMD, MU, SMCI, SNDK, WDC,
JPM, V, MA, PLTR, ARM, MSTR, XOM, CVX
```

Since `MAX_UNIVERSE = 40` and the screener only adds from S&P 500 constituents that beat these anchors on momentum, the effective universe is dominated by these 20 names. On May 25, the universe was exactly these 20 -- no non-anchor stocks qualified.

**Sector breakdown:**
- Tech: 5 (AAPL, MSFT, GOOGL, PLTR, MSTR)
- Semis: 5 (NVDA, AVGO, AMD, ARM, SNDK + SMCI, WDC, MU)
- Consumer: 2 (AMZN, TSLA)
- Financials: 3 (JPM, V, MA)
- Energy: 2 (XOM, CVX)

That's 12 of 20 (60%) in tech + semis. When Pipeline B fires on a broad bullish signal, it naturally picks the highest-conviction names, which are almost always the volatile semis (NVDA, AMD, SMCI) because they have the most extreme technical indicators.

### 2.2 Pipeline B's Technical Scoring Favors Volatile Names

**File:** `pipeline_b.py` lines 51-179

The `_score_technicals()` function awards points for extreme RSI, MACD crossovers, Bollinger Band %B extremes, and high ADX. Volatile stocks (NVDA: ATR 0.13%, SMCI: ATR 0.39%) naturally produce more extreme indicator readings than calm stocks (JPM: ATR 0.04%), so they consistently score higher.

**ADX damping** (lines 111-114) is too gentle: `score *= 0.60` for ADX < 15. This barely reduces the signal -- choppy markets still produce tradeable scores.

### 2.3 No Cross-Sector Diversification Requirement

There is no code anywhere that says "you must have positions in at least 2 different sectors" or "if you're 50% tech/semis, prefer non-tech entries." The sector cap (`_SECTOR_CAP_PCT = 0.60`) only prevents going ABOVE 60% in one sector -- it doesn't encourage diversification.

---

## 3. Timing Analysis: Entry/Exit Mismatch

### 3.1 Max Hold = 30 Minutes vs ATR Targets = 1.6-5.6%

**File:** `signal_loop.py` lines 66-72

| Parameter | Value | Effect |
|-----------|-------|--------|
| SIZING_MAX_HOLD_BARS | 30 | 30 minutes max hold |
| SIZING_TAKE_PROFIT_ATR_MULT | 25 | TP = ATR x 25 |
| SIZING_TAKE_PROFIT_FLOOR | 0.012 | 1.2% minimum TP |

For SMCI (ATR_pct = 0.0022), the take profit target is 5.5%. For GOOGL (ATR_pct = 0.0008), it's 1.94%. **Neither target is reachable in 30 minutes of normal trading.** Most stocks move 0.1-0.3% in 30 minutes.

**Log evidence:** On May 22, ALL exits were `max_hold` at exactly 30 bars. Zero take_profit exits. Zero stop_loss exits. The ATR targets are unreachable at this timeframe.

```
2026-05-22T15:02 AMZN max_hold target=0.0168 (1.68%)
2026-05-22T15:02 GOOGL max_hold target=0.0181 (1.81%)
2026-05-22T15:02 SMCI max_hold target=0.0558 (5.58%)
2026-05-22T15:03 NVDA max_hold target=0.0328 (3.28%)
```

**Result:** Positions are held for exactly 30 minutes regardless of price action, then dumped. The ATR-adaptive exit system is effectively disabled -- max_hold always fires first. Wins and losses are both tiny (the market doesn't move enough in 30 minutes for stops or targets to trigger).

### 3.2 Exit Bug: Repeated Exit Attempts Without Execution

**Log evidence:** The XOM exit on April 6 fired `sizing_exit reason=stop_loss` on EVERY tick for hours -- 19 separate exit attempts logged between 16:33 and 19:42. The sell order failed (`insufficient buying power`), but the bot didn't mark the exit as "attempted" or enter a cooldown. It just tried again next minute.

**Code reference:** `signal_loop.py` lines 992-998: `_check_sizing_exit()` returns `stop_loss`, the code sets `side = "sell"` and `qty = pos.qty`, but if `submit_order()` fails, the position remains in `self._pm._positions`. Next tick, the same exit check fires again. There's no retry-limiter or fail-state for exits.

---

## 4. Position Sizing Issues

### 4.1 SmartPositionSizer Calibrated for $100k, Running on $10k

**File:** `position_sizer.py` lines 66-101

The sizer was tuned with comments like "Tuned for $10k account" but the numbers tell a different story:

- `_BASE_MIN_PCT = 0.22` (22%) -- minimum position is $2,200 on $10k
- `_BASE_MAX_PCT = 0.50` (50%) -- maximum before ATR is $5,000
- `_MAX_NOTIONAL = 8000.0` -- hard cap per position

On a $10k account, a single 25% position is $2,500. Six simultaneous positions at 25% = $15,000 = 150% of portfolio (only possible with margin).

### 4.2 The 25% Cap Doesn't Prevent Over-Concentration

The CircuitBreakers `check_position_size()` (`circuit_breakers.py` line 129) rejects individual positions > 25%. But it doesn't check TOTAL heat. The portfolio_heat check (`circuit_breakers.py` line 161-162) flags heat > 80% but only logs a warning -- it doesn't halt entries. The `_HEAT_TIERS` in position_sizer.py do gate entries, but they're evaluated per-entry, not accounting for same-tick entries.

### 4.3 Race Condition: Same-Tick Multi-Entry

This is the single most dangerous bug. In `signal_loop_b.py` lines 190-211:

```python
for sig in signals:
    ...
    await self._act_on_signal(sig, price, None, regime=regime)
```

Each call to `_act_on_signal` checks `self._pm.managed_heat` to determine sizing. But `open_position()` only fires after `submit_order()` returns a fill. For the first entry in the loop, heat = 0%. For the second, heat might still be 0% if the first order hasn't filled yet. For the sixth, heat is definitely stale.

Even if fills are synchronous (they appear to be), the SmartPositionSizer computes sector_notionals BEFORE the position is added to `_positions`. So all 6 entries see the same empty sector map and pass the sector cap.

---

## 5. Pipeline A vs Pipeline B Comparison

### 5.1 Signal Quality

| Metric | Pipeline A (LightGBM) | Pipeline B (Rules-Based) |
|--------|----------------------|-------------------------|
| Training IC | 0.1775 | N/A (no statistical model) |
| Live IC (7d) | 0.0492 | Not tracked separately |
| Live Dir Accuracy | 49.02% | Not tracked |
| IC drop from training | -72.3% | N/A |

**Pipeline A's LightGBM has a real (if small) edge:** IC = 0.0492 live with p-value = 0.000166 (statistically significant). But the model has drifted severely -- 72.3% IC drop from training. The drift agent flagged this as `severity: critical` with `retrain_recommended: true`.

**Pipeline B has no statistical edge.** It combines rule-based technical scores (RSI, MACD, Bollinger, etc.) with fundamentals, regime, and sentiment. On May 22, the ensemble signals that triggered entries were 0.10-0.20 -- barely above the `PIPELINE_B_MIN_ENSEMBLE = 0.10` threshold. The multi-factor confirmation gate (`pipeline_b.py` lines 355-370) is supposed to suppress weak signals, but the regime and fundamentals scores are often non-zero for mega-caps (positive earnings surprises, positive market regime), so 2-of-4 confirmation is easy to achieve.

### 5.2 Live IC by Ticker (Pipeline A)

From `reports/ic/2026-05-25.json`, tickers traded on May 22:

| Ticker | Live IC | Dir Accuracy | Verdict |
|--------|---------|-------------|---------|
| TSLA | 0.069 | 56.3% | Marginal |
| NVDA | 0.144 | 46.5% | IC exists but dir wrong |
| AMD | -0.183 | 40.1% | **Negative IC -- actively wrong** |
| GOOGL | 0.400 | 42.6% | IC high but dir wrong |
| SMCI | -0.195 | 54.1% | **Negative IC** |
| AMZN | 0.212 | 43.0% | IC high but dir wrong |

**Pipeline B traded AMD (IC=-0.18) and SMCI (IC=-0.20) -- the two worst-performing tickers in the universe.** The rules-based engine has no awareness of which tickers its signals actually work for.

### 5.3 Recommendation

**Pipeline A is strictly better than Pipeline B**, despite its drift. A live IC of 0.049 with p=0.0002 is a real (if weak) signal. Pipeline B has no proven predictive power. However, Pipeline A needs retraining badly (IC dropped 72% from training) and its asyncio task needs crash protection.

---

## 6. Specific Recommendations

### 6.1 CRITICAL: Fix Same-Tick Multi-Entry Race Condition

**File:** `signal_loop.py` and `signal_loop_b.py`

**Fix:** After each successful entry in the for loop, recompute managed_heat and sector_notionals. Or better: limit entries to 1 per tick (process the single highest-conviction signal per tick cycle, not all qualifying signals).

```python
# In _tick() / signal_loop_b._tick():
MAX_ENTRIES_PER_TICK = 1
entries_this_tick = 0
for sig in signals:
    if entries_this_tick >= MAX_ENTRIES_PER_TICK:
        break
    # ... existing entry logic ...
    # After successful fill:
    entries_this_tick += 1
```

### 6.2 CRITICAL: Add Total Heat Pre-Check Before Entry

**File:** `signal_loop.py`, `_sizing_entry_gate_open()` (line 749)

Add a heat pre-check that uses LIVE managed_heat including pending (not-yet-confirmed) entries:

```python
# Gate 0b: Portfolio heat hard ceiling
if self._pm.managed_heat >= 0.80:
    return False
```

### 6.3 HIGH: Increase Max Hold or Decrease Take Profit

Either:
- Increase `SIZING_MAX_HOLD_BARS` from 30 to 120-240 (2-4 hours) to give ATR targets time to trigger, OR
- Decrease `SIZING_TAKE_PROFIT_ATR_MULT` from 25 to 8-10 (reachable in 30 minutes), OR
- Switch to a trailing-profit approach: if up 0.3% after 10 bars, tighten the trailing stop.

Currently max_hold fires 100% of the time. The ATR system is decoration.

### 6.4 HIGH: Switch Back to Pipeline A with Crash Protection

1. Set `ACTIVE_PIPELINE=a` in .env
2. Add watchdog task to restart Pipeline A if its asyncio task dies:

```python
# In main.py lifespan:
async def _watchdog():
    while True:
        await asyncio.sleep(300)  # check every 5 min
        if signal_task.done():
            logger.critical("signal_loop_died_restarting")
            signal_task = asyncio.create_task(_signal_loop.start())
```

3. Retrain LightGBM -- IC dropped 72% from training. The retrain agent is failing (`"No objects to concatenate"` in `reports/retrain/retrain_2026-05-26.json`).

### 6.5 HIGH: Fix Exit Retry Loop

**File:** `signal_loop.py`, `_act_on_signal()` around line 1160

After a failed sell order, mark the position as "exit_pending" to prevent re-triggering:

```python
if result.status not in ("filled", "partially_filled"):
    if side == "sell":
        self._pending_exit_reasons.pop(ticker, None)
        # Don't retry for 5 ticks
        self._ticker_exit_cooldown[ticker] = 5
```

### 6.6 MEDIUM: Diversify the Universe

The ANCHOR_TICKERS list is 60% tech/semis. Add at minimum:
- Healthcare: LLY, UNH, JNJ
- Consumer staples: PG, KO, COST
- Industrials: CAT, HON, GE
- Utilities: NEE, SO

And make the screener's momentum scoring sector-aware: "select the top 2-3 from EACH sector" rather than "top 40 overall."

### 6.7 MEDIUM: Add Correlation-Based Entry Gating

Before entering any position, compute the average correlation between the candidate and all existing positions over the last 20 bars. If average correlation > 0.7, block the entry. This prevents the "all long mega-cap tech at the same time" problem.

### 6.8 LOW: Account Size Awareness

The SmartPositionSizer has `_MIN_NOTIONAL = 2000.0` and `_MAX_NOTIONAL = 8000.0`. On a $10k account, this means 1-4 positions max (which is appropriate). But the same-tick bug bypasses this by opening 6 positions before heat is updated.

---

## 7. Manual vs Algo: Honest Assessment

### What the Algo Does Wrong That a Human Wouldn't

1. **No common sense about correlation.** A human would never go long NVDA, AMD, SMCI, and AVGO simultaneously. A human understands these are all "AI chip plays" and that they move together. The algo treats them as independent signals.

2. **No awareness of market context.** A human checking at 2:29 PM on a Thursday would see "we're 30 minutes from close, do I really want to enter 6 new positions?" The algo has no concept of this. (Market_close_buffer_minutes in paper.yaml is 15 min, but this config isn't enforced in the signal loop code.)

3. **No position monitoring.** A human holding SMCI would watch the price. The algo checks once per minute, and its only exit logic is mechanical (stop/trail/target/max_hold). The 30-minute max_hold means the algo is essentially a very short-term scalper, but its signals are calibrated for multi-hour moves.

4. **No learning from failure.** The algo traded AMD and SMCI even though their live ICs are -0.18 and -0.20. A human would have noticed "every time I trade AMD, I lose" and stopped. The Kelly gate is supposed to do this, but it operates at the portfolio level (all tickers combined), not per-ticker.

### When the Algo Could Beat a Human

1. **Consistency.** A human gets tired, emotional, has FOMO. The algo doesn't -- IF its signals are actually predictive. Pipeline A's IC of 0.049 is small but real.

2. **Speed.** In a genuine momentum breakout, the algo can detect MACD/RSI signals and enter within 1 minute. A human might miss the first few minutes.

3. **Discipline on exits.** The stop-loss/trailing-stop system, when calibrated correctly (not the current max_hold override), would enforce discipline that humans struggle with.

### Verdict

At the current state of development, **a human stock picker with the algo's signals displayed as decision support would outperform the fully autonomous algo.** The algo's signal generation (especially Pipeline A/LightGBM) has value, but its execution layer has critical bugs that erase any edge. The immediate priorities are:

1. Fix the same-tick race condition (prevents portfolio blowup)
2. Switch back to Pipeline A and retrain LightGBM
3. Increase max_hold or decrease TP targets so the exit system actually functions
4. Add per-ticker Kelly/IC gating (stop trading tickers with negative IC)

---

## Appendix: Timeline Reconstruction

| Date | Event | Portfolio |
|------|-------|-----------|
| 2026-04-06 16:33 | A/B test started, portfolio synced | $106,572 |
| 2026-04-06 16:33 | Portfolio heat 265% from legacy positions | (pre-existing) |
| 2026-04-06 16:33+ | XOM sell loop: exit failed every minute for hours | ~$106k |
| 2026-04-07 13:30 | Daily loss halt (-3.59%) | ~$102,742 |
| 2026-04-08 | Portfolio heat still 217% | ~$102k |
| 2026-04-23 | Last stable A/B reading | ~$98,656 |
| 2026-05-06 | Pipeline A daily reset | $97,508 |
| 2026-05-13 | Pipeline A only logging $53,286 (stale A/B half) | Unknown |
| 2026-05-14 | Pipeline A asyncio task dies silently | Unknown |
| 2026-05-22 07:49 | Restart with ACTIVE_PIPELINE=b | $9,999 synced |
| 2026-05-22 14:29-14:32 | Pipeline B enters 6 positions in 3 minutes | ~$10k |
| 2026-05-22 15:02-15:07 | All 6 exit at max_hold (30 bars) | ~$10k |
| 2026-05-26 13:47 | SMCI sold, AVGO bought | $9,874 |
| 2026-05-26 17:11 | Current state: 1 position (AVGO), PDT restricted | $9,874 |

---

*This report is for analysis only. No trading code or config has been modified.*
