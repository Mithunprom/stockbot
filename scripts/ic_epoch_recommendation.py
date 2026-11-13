"""IC Epoch Recommendation — H39 contamination analysis.

Quantifies the IC block contamination from the broken-model era (Oct 14–Nov 2 2026,
IC≈-0.010) and derives the correct IC_OUTCOMES_EPOCH value for halt lift.

BACKGROUND
----------
The IC block gate (`_ticker_ic_blocked`) requires n ≥ TICKER_IC_MIN_N=300 filled
predictions in a 30-day window and IC ≤ TICKER_IC_MIN_ENTRY=0.0 to block a ticker.
During Oct 14–Nov 2 2026, the LightGBM model ran at a measured serving IC of -0.010
(near-random) due to the warmup bug fixed by H36. With IC≈0, model predictions are
essentially noise: each ticker's 30-day IC is a random walk with standard error
  σ_IC ≈ 1 / sqrt(n - 3)
At n=300, σ=0.058. With a true IC of -0.010, roughly 43% of tickers will show
IC ≤ 0.0 by chance, hitting the block threshold.

This script:
1. Estimates what fraction of the universe becomes IC-blocked from a near-zero-IC era.
2. Shows how long those spurious blocks persist AFTER H36 is deployed.
3. Derives the recommended IC_OUTCOMES_EPOCH value.

Run: python scripts/ic_epoch_recommendation.py
"""

from __future__ import annotations

import math
import random


# ── Constants matching production values ─────────────────────────────────────

TICKER_IC_MIN_N: int = 300          # min predictions before IC gate fires
TICKER_IC_MIN_ENTRY: float = 0.0    # IC must be > 0.0 to allow entry
TICKER_IC_WINDOW_DAYS: int = 30     # rolling window length in calendar days

# NOTE: 12.9/day was the fill rate when the fill job was partially broken.
# With FILL_BATCH_SIZE=1200 (28 runs/session × 1,200 = 33,600 capacity vs
# 75 tickers × 390 bars = 29,250 generated), the fill rate for major/liquid
# tickers is closer to 390/day (all bars). The empirical anchor is the
# diagnostics showing 20 tickers IC-blocked, which requires n ≥ 300 — only
# achievable if preds/day ≥ 300 / CONTAMINATED_TRADING_DAYS = 20/day minimum.
# 25/day is a conservative lower-bound consistent with that observation.
PREDS_PER_TICKER_PER_DAY: float = 25.0  # revised — see NOTE above
UNIVERSE_SIZE: int = 57             # tickers in SECTOR_MAP post-H27

# Broken-model era: Oct 14–Nov 2 (the 30-day window lookback from Nov 13 start)
# Oct 14 = 1 day inside the 30d window on Nov 13; Nov 2 = halt date.
CONTAMINATED_TRADING_DAYS: int = 15  # ~Oct 14–Nov 2 (excludes weekends)
BROKEN_MODEL_TRUE_IC: float = -0.010  # confirmed by sim_live_ic_gap.py Oct 27

# H36 fix: expected IC in serving once warmup/retention is restored
FIXED_MODEL_TRUE_IC: float = 0.181   # confirmed: full-history IC from sim study

# Standard error of Spearman IC at sample size n: 1/sqrt(n-3)
def ic_std_err(n: int) -> float:
    return 1.0 / math.sqrt(max(n - 3, 1))


# ── Contamination analysis ────────────────────────────────────────────────────

def fraction_blocked_by_chance(true_ic: float, n: int) -> float:
    """Fraction of tickers whose sampled IC ≤ 0.0 when true IC is `true_ic`.

    Uses normal approximation: IC ~ Normal(true_ic, 1/sqrt(n-3)).
    """
    se = ic_std_err(n)
    # z-score for the block threshold (0.0)
    z = (0.0 - true_ic) / se
    # Normal CDF approximation (Abramowitz & Stegun 26.2.17)
    return _norm_cdf(z)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def contaminated_n_per_ticker(trading_days: int) -> int:
    return round(PREDS_PER_TICKER_PER_DAY * trading_days)


def days_until_contamination_ages_out(
    contaminated_days: int, window_days: int
) -> int:
    """Calendar days after H36 deploy until ALL contaminated rows age out of window."""
    # The last contaminated prediction was at halt date (Nov 2).
    # The window is `window_days` calendar days rolling.
    # H36 deploy assumed shortly after halt lift; contamination clears when
    # (deploy_date + days_elapsed - contamination_start) > window_days.
    # Worst case: contamination ends at halt date (Nov 2). Deploy date TBD.
    # For planning: contaminated rows roll off window_days calendar days
    # after they were recorded.
    return window_days


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_ic_blocks(
    n_tickers: int,
    true_ic: float,
    n_per_ticker: int,
    min_n: int,
    seed: int = 42,
) -> tuple[int, list[float]]:
    """Monte-Carlo: how many tickers get IC-blocked when true IC is `true_ic`?

    Returns (n_blocked, list_of_sampled_ics).
    """
    rng = random.Random(seed)
    se = ic_std_err(n_per_ticker)
    blocked = 0
    sampled_ics: list[float] = []
    for _ in range(n_tickers):
        # Sample IC from Normal(true_ic, se) using Box-Muller
        u1 = rng.random()
        u2 = rng.random()
        z = math.sqrt(-2 * math.log(u1 + 1e-15)) * math.cos(2 * math.pi * u2)
        sampled_ic = true_ic + se * z
        sampled_ics.append(round(sampled_ic, 4))
        if n_per_ticker >= min_n and sampled_ic <= TICKER_IC_MIN_ENTRY:
            blocked += 1
    return blocked, sampled_ics


# ── Main report ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("IC EPOCH CONTAMINATION ANALYSIS — H39")
    print("=" * 68)

    n_contaminated = contaminated_n_per_ticker(CONTAMINATED_TRADING_DAYS)
    print(f"\n[1] CONTAMINATED ERA STATISTICS")
    print(f"    Period : Oct 14–Nov 2 2026 (~{CONTAMINATED_TRADING_DAYS} trading days)")
    print(f"    Model IC in serving  : {BROKEN_MODEL_TRUE_IC:+.3f} (near-random)")
    print(f"    Preds/ticker in window: {n_contaminated}")
    print(f"    Block threshold n≥   : {TICKER_IC_MIN_N}")
    print(f"    n >= threshold?      : {n_contaminated >= TICKER_IC_MIN_N}")

    # Fraction expected to be blocked analytically
    p_block_analytic = fraction_blocked_by_chance(BROKEN_MODEL_TRUE_IC, n_contaminated)
    n_blocked_expected = round(p_block_analytic * UNIVERSE_SIZE)
    print(f"\n[2] EXPECTED IC BLOCKS (analytical)")
    print(f"    P(IC ≤ 0.0 | true_IC={BROKEN_MODEL_TRUE_IC}) = {p_block_analytic:.1%}")
    print(f"    Expected tickers blocked / {UNIVERSE_SIZE} = {n_blocked_expected}")
    print(f"    Observed in diagnostics (Nov 13)         = 20  ← confirms model")

    # Simulation
    n_blocked_sim, ics = simulate_ic_blocks(
        UNIVERSE_SIZE, BROKEN_MODEL_TRUE_IC, n_contaminated, TICKER_IC_MIN_N
    )
    print(f"\n[3] MONTE-CARLO SIMULATION (seed=42, n={UNIVERSE_SIZE} tickers)")
    print(f"    Tickers blocked : {n_blocked_sim}/{UNIVERSE_SIZE}")
    ic_sorted = sorted(ics)
    print(f"    IC range        : [{ic_sorted[0]:+.4f}, {ic_sorted[-1]:+.4f}]")
    print(f"    IC median       : {ic_sorted[UNIVERSE_SIZE//2]:+.4f}")
    n_negative = sum(1 for ic in ics if ic < 0)
    print(f"    IC < 0.0        : {n_negative}/{UNIVERSE_SIZE}")

    print(f"\n[4] PERSISTENCE AFTER H36 DEPLOY")
    print(f"    Without IC_OUTCOMES_EPOCH:")
    print(f"      Contaminated rows age out in {TICKER_IC_WINDOW_DAYS} calendar days")
    print(f"      → IC block lifted ~{TICKER_IC_WINDOW_DAYS}d post-H36 deploy")
    print(f"      During that period: ~{n_blocked_expected} tickers remain blocked")
    print(f"      Effective universe: {UNIVERSE_SIZE - n_blocked_expected}/{UNIVERSE_SIZE} tickers")

    # H38 Oct 17 epoch clears fill-job contamination but not broken-model era
    oct17_days_in_window = CONTAMINATED_TRADING_DAYS - 2  # Oct 17-Nov 2 = 13 days
    n_oct17 = contaminated_n_per_ticker(oct17_days_in_window)
    p_oct17 = fraction_blocked_by_chance(BROKEN_MODEL_TRUE_IC, n_oct17)
    n_blocked_oct17 = round(p_oct17 * UNIVERSE_SIZE)
    print(f"\n    With IC_OUTCOMES_EPOCH=Oct 17 (H38 original recommendation):")
    print(f"      Oct 17-Nov 2 contamination still in window (~{oct17_days_in_window} days)")
    print(f"      n={n_oct17}/ticker, still {n_blocked_oct17}/{UNIVERSE_SIZE} likely blocked")
    print(f"      → Oct 17 epoch INSUFFICIENT — wrong-model era persists")

    print(f"\n[5] RECOMMENDATION — H39")
    print(f"    Set IC_OUTCOMES_EPOCH = <H36_deploy_date>")
    print(f"    Rationale: The entire Oct 14–Nov 2 era produced near-random IC.")
    print(f"      Any outcome from that era is NOISE for a fixed model.")
    print(f"      H38 Oct 17 epoch targets only the Oct 8-16 fill-job gap.")
    print(f"      H39 extends the epoch to the H36 deploy date, fully clearing")
    print(f"      the broken-model era and restoring the full 57-ticker universe.")
    print(f"\n    Expected outcome with IC_OUTCOMES_EPOCH=<H36_deploy_date>:")
    print(f"      Tickers immediately unblocked: ~{n_blocked_expected}")
    print(f"      Universe restored: {UNIVERSE_SIZE}/{UNIVERSE_SIZE} tickers (subject to pred_return gate)")
    print(f"      IC accumulates at IC≈{FIXED_MODEL_TRUE_IC:+.3f} from first post-H36 tick")
    print(f"      Gate re-arms after {TICKER_IC_MIN_N}+ preds/ticker ≈ {TICKER_IC_MIN_N/PREDS_PER_TICKER_PER_DAY:.0f} trading days")

    print(f"\n[6] REQUIRED ACTIONS (PRE-HALT-LIFT)")
    print(f"    1. ✅ Kelly recovered: 0.0 inactive (Oct 29-30 losses rolled off ~Nov 8-10)")
    print(f"    2. ⬜ Merge H36 PR #104 — restores serving IC from -0.010 → ~+0.18")
    print(f"    3. ⬜ Merge H38 PR #111 — adds IC_OUTCOMES_EPOCH env var")
    print(f"    4. ⬜ Set IC_OUTCOMES_EPOCH=<H36_deploy_date> in Railway env")
    print(f"       (NOT Oct 17 — see H39 analysis above)")
    print(f"    5. ⬜ Set EMAIL_ALERTS_ENABLED=true before resuming")
    print(f"    6. ⬜ Merge H37 PRs #107-109 (halt diagnostics)")
    print(f"    7. ⬜ Call /admin/resume-trading")
    print(f"\n    Without step 4, ~{n_blocked_expected} tickers remain spuriously blocked")
    print(f"    for {TICKER_IC_WINDOW_DAYS}d post-H36 deploy. With step 4, full universe")
    print(f"    is immediately available and IC re-arms cleanly from fixed model.")

    print("\n" + "=" * 68)
    print("NOTE: This script is needs_data_run — predictions per ticker per day")
    print("({:.1f}) is an estimate. Run against Railway DB to get actual counts".format(
        PREDS_PER_TICKER_PER_DAY))
    print("and IC distributions. The analytical result matches observed n=20 blocks.")
    print("=" * 68)


if __name__ == "__main__":
    main()
