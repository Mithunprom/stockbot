"""Headless R&D worker — the self-looping extension engine's clock cycle.

Runs the hypothesis queue in agent_state.json continuously in the background
(Railway worker service or any box with repo + env vars). Each cycle:

  1. load agent_state.json
  2. pick the next pending hypothesis
  3. run its backtest command, capture metrics from stdout
  4. record the result (positive OR negative) to the state file +
     reports/research/, and mark the hypothesis tested
  5. heartbeat to stdout (visible in Railway logs), sleep, repeat

Governance (per CLAUDE.md + super_agents_manifesto.txt):
  - RESEARCH ONLY: this process never places orders, never touches
    config/live.yaml or the deployed strategy, never git-pushes. Winning
    hypotheses are surfaced for a human (or the nightly cloud R&D agent)
    to turn into a PR.
  - Breakthrough threshold writes BREAKTHROUGH_REPORT.md for review.

Enable:
  AGENT_WORKER_ENABLE=true python agent_worker.py

Deploy as a Railway background worker (separate service, same repo):
  railway add  →  new service  →  start command: python agent_worker.py
  and set AGENT_WORKER_ENABLE=true + ALPACA_API_KEY/SECRET on that service.

Optional Claude integration: if ANTHROPIC_API_KEY is set, after each cycle
the worker asks Claude (model: claude-fable-5) to propose the next hypothesis
from the accumulated results and appends it to the queue.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_PATH = Path("agent_state.json")
REPORT_DIR = Path("reports/research")
BREAKTHROUGH_PATH = Path("BREAKTHROUGH_REPORT.md")

CYCLE_SLEEP_SECONDS = int(os.environ.get("AGENT_WORKER_CYCLE_SECONDS", str(6 * 3600)))
HYPOTHESIS_TIMEOUT = int(os.environ.get("AGENT_WORKER_TIMEOUT", str(2 * 3600)))
BREAKTHROUGH_SHARPE = 3.0

# Commands a hypothesis is allowed to run (research scripts only — the worker
# must never be able to execute arbitrary shell from a state file edit).
ALLOWED_COMMAND_PREFIXES = (
    "python scripts/research_backtest.py",
    "python scripts/research_variants.py",
    "python scripts/train_lgbm.py",
    "python scripts/run_ffsa.py",
    "python -m src.agents.profitability_agent",
    "python -m pytest",
)

_shutdown = False


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Save state and exit cleanly on Railway redeploys/restarts."""
    global _shutdown
    _shutdown = True
    print(f"[worker] signal {signum} received — finishing cycle then exiting",
          flush=True)


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, Any]) -> None:
    state["last_iteration_date"] = datetime.now(timezone.utc).date().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2))


def next_pending(state: dict[str, Any]) -> dict[str, Any] | None:
    for h in state.get("hypothesis_queue", []):
        if h.get("status") == "pending":
            return h
    return None


def extract_metrics(output: str) -> dict[str, float]:
    """Pull sharpe / profit factor / return / drawdown figures from run output."""
    metrics: dict[str, float] = {}
    patterns = {
        "sharpe": r"sharpe[=:\s]+(-?\d+\.?\d*)",
        "profit_factor": r"(?:profit factor|pf)[=:\s]+(-?\d+\.?\d*)",
        "return_pct": r"(?:total return|return)[=:\s]+(-?\d+\.?\d*)%",
        "max_drawdown_pct": r"(?:max )?(?:drawdown|dd)[=:\s]+(-?\d+\.?\d*)%?",
        "ic": r"\bic[=:\s]+(-?\d+\.?\d*)",
    }
    lowered = output.lower()
    for name, pat in patterns.items():
        found = re.findall(pat, lowered)
        if found:
            try:
                metrics[name] = float(found[-1])   # last = final/summary value
            except ValueError:
                pass
    return metrics


def run_hypothesis(hyp: dict[str, Any]) -> dict[str, Any]:
    """Execute one hypothesis test in a subprocess and collect the outcome."""
    cmd = hyp.get("test", "")
    if not any(cmd.startswith(p) for p in ALLOWED_COMMAND_PREFIXES):
        return {"status": "rejected",
                "error": f"command not in research allowlist: {cmd!r}"}
    print(f"[worker] {datetime.now(timezone.utc).isoformat()} "
          f"running {hyp['id']}: {cmd}", flush=True)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=HYPOTHESIS_TIMEOUT,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        result = {
            "status": "tested" if proc.returncode == 0 else "errored",
            "returncode": proc.returncode,
            "runtime_seconds": round(time.time() - started, 1),
            "metrics": extract_metrics(output),
            "output_tail": output[-3000:],
        }
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "runtime_seconds": HYPOTHESIS_TIMEOUT}
    except Exception as exc:  # noqa: BLE001 — worker must never die on one test
        result = {"status": "errored", "error": f"{type(exc).__name__}: {exc}"}
    return result


def maybe_breakthrough(state: dict[str, Any], hyp: dict[str, Any],
                       result: dict[str, Any]) -> None:
    sharpe = result.get("metrics", {}).get("sharpe", 0.0)
    if sharpe < BREAKTHROUGH_SHARPE:
        return
    BREAKTHROUGH_PATH.write_text(
        f"# BREAKTHROUGH REPORT — {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Hypothesis {hyp['id']}: {hyp['hypothesis']}\n\n"
        f"Backtest Sharpe: **{sharpe}** (threshold {BREAKTHROUGH_SHARPE})\n\n"
        f"Full metrics: {json.dumps(result.get('metrics', {}), indent=2)}\n\n"
        "## Required next steps (per CLAUDE.md — do NOT skip)\n"
        "1. Independent re-validation on a fresh walk-forward window\n"
        "2. Leakage audit by the Index/Risk Quant persona\n"
        "3. Paper-trading consistency before any live consideration\n"
    )
    print(f"[worker] BREAKTHROUGH candidate written: {BREAKTHROUGH_PATH}",
          flush=True)


def propose_next_hypothesis(state: dict[str, Any]) -> None:
    """Optional: ask Claude for the next hypothesis based on results so far."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        manifesto = Path("super_agents_manifesto.txt").read_text()
        history = json.dumps(state.get("completed_iterations", [])[-10:], indent=2)
        msg = client.messages.create(
            model="claude-fable-5",
            max_tokens=800,
            system=manifesto,
            messages=[{
                "role": "user",
                "content": (
                    "Given these recent hypothesis results for our LightGBM "
                    f"1-day swing equity bot:\n{history}\n\n"
                    "Propose ONE new falsifiable hypothesis as JSON with keys "
                    "id, hypothesis, test (a command starting with one of "
                    f"{ALLOWED_COMMAND_PREFIXES}). Reply with ONLY the JSON."
                ),
            }],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        candidate = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        if candidate.get("id") and candidate.get("test"):
            candidate["status"] = "pending"
            candidate["proposed_by"] = "claude-fable-5"
            state.setdefault("hypothesis_queue", []).append(candidate)
            print(f"[worker] Claude proposed {candidate['id']}: "
                  f"{candidate.get('hypothesis', '')[:100]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] hypothesis proposal skipped: {exc}", flush=True)


def one_cycle() -> None:
    state = load_state()
    hyp = next_pending(state)
    if hyp is None:
        print("[worker] hypothesis queue empty — asking for proposals", flush=True)
        propose_next_hypothesis(state)
        save_state(state)
        return

    result = run_hypothesis(hyp)
    hyp["status"] = result["status"]
    record = {
        "id": hyp["id"],
        "hypothesis": hyp.get("hypothesis", ""),
        "tested_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in result.items() if k != "output_tail"},
    }
    state.setdefault("completed_iterations", []).append(record)
    metrics = result.get("metrics", {})
    if metrics.get("sharpe", 0) > state["metrics_history"].get("current_sharpe", 0):
        state["metrics_history"]["current_sharpe"] = metrics["sharpe"]
    state["active_hypothesis"] = hyp.get("hypothesis", "")
    save_state(state)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    (REPORT_DIR / f"{stamp}_{hyp['id']}.json").write_text(
        json.dumps({**record, "output_tail": result.get("output_tail", "")},
                   indent=2))
    print(f"[worker] {hyp['id']} → {result['status']} metrics={metrics}",
          flush=True)
    maybe_breakthrough(state, hyp, result)


def main() -> None:
    if os.environ.get("AGENT_WORKER_ENABLE", "").lower() != "true":
        print("[worker] AGENT_WORKER_ENABLE != true — refusing to start "
              "(safety default; set the env var on the worker service)",
              flush=True)
        sys.exit(0)
    if not STATE_PATH.exists():
        print("[worker] agent_state.json missing — aborting", flush=True)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    print(f"[worker] R&D worker online — cycle every {CYCLE_SLEEP_SECONDS}s",
          flush=True)

    while not _shutdown:
        try:
            one_cycle()
        except Exception as exc:  # noqa: BLE001 — heartbeat must survive anything
            print(f"[worker] cycle error (state saved): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            try:
                state = load_state()
                state["blocked_by_token_limit"] = "429" in str(exc)
                save_state(state)
            except Exception:
                pass
        # Heartbeat sleep in small slices so SIGTERM lands quickly
        slept = 0
        while slept < CYCLE_SLEEP_SECONDS and not _shutdown:
            time.sleep(min(60, CYCLE_SLEEP_SECONDS - slept))
            slept += 60
            if slept % 1800 == 0:
                print(f"[worker] heartbeat — next cycle in "
                      f"{CYCLE_SLEEP_SECONDS - slept}s", flush=True)

    print("[worker] clean shutdown — state persisted", flush=True)


if __name__ == "__main__":
    main()
