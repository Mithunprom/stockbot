#!/usr/bin/env python3
"""Next-session forecast for a single ticker (default SNDK).

Thin CLI wrapper over src.analysis.forecast (the shared implementation used by
the daily ForecastEmailAgent). Writes reports/forecasts/{TICKER}_{DATE}.json and
prints a human-readable summary.

Usage:
    python scripts/forecast_sndk.py [TICKER]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.forecast import build_forecast, render_text, save_forecast


def main() -> int:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "SNDK").upper()
    fc = build_forecast(ticker)
    out_path = save_forecast(fc)
    print(render_text(fc))
    print(f"\n[written] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
