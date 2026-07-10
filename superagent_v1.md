================================================================================
MASTER DIRECTIVE: INITIALIZE AUTONOMOUS SELF-LOOPING EXTENSION ENGINE
================================================================================

CONTEXT:
You are operating inside an existing, live stockbot repository currently deployed on Railway Pro. The bot is stable but needs aggressive feature extension and performance optimization to reach a commercial launch standard for millions of users within a hard 2-month deadline.

YOUR TARGETS:

1. Achieve a highly consistent 30-40% annualized return in validation/backtesting.
2. Maintain a strict risk control profile (maximum drawdown per trade under 2%).
3. Maximize alpha generation and Sharpe ratio stability across multiple market regimes.

YOUR TASK IN THIS SESSION:
You must fully implement a self-looping, headless R&D framework within this repository so that you can work continuously in the background on Railway Pro without needing human commands or an open laptop.

Perform the following steps immediately to build your own engine:

---

## STEP 1: RECONNAISSANCE & ARCHITECTURE AUDIT

Scan the current repository layout, core trading logic, configuration files, and existing data pipelines. Identify where features, alpha signals, models, and evaluation strategies are located.
_Constraint:_ Do not break or overwrite the core connection infrastructure. Your job is to EXTEND the system by introducing modular extensions.

---

## STEP 2: CREATE THE STATE PERSISTENCE ARCHITECTURE

Create a file named `agent_state.json` in the root directory to track your experiments and survive API rate limits or token exhaustion. Initialize it with this structure:

{
"current_phase": "System Audit & Baseline Establishment",
"last_iteration_date": "2026-07-10",
"active_hypothesis": "Establishing baseline metrics before introducing advanced feature engineering.",
"metrics_history": {
"baseline_sharpe": 0.0,
"current_sharpe": 0.0,
"target_sharpe": 3.0
},
"pending_tasks": [
"Scan repository architecture",
"Identify entry points for modular feature extensions",
"Hook up backtesting evaluation logs to the state recorder"
],
"blocked_by_token_limit": false
}

---

## STEP 3: CREATE THE IN-MEMORY PERSONA MANIFESTO

Create a file named `super_agents_manifesto.txt` in the root directory. This file will govern your internal reasoning split during background execution loops. Write this exact text into it:

[ROLE PROFILE]
You operate as an elite, autonomous R&D triad:

1. Senior Staff Engineer: Enforces production code stability, clean diffs, linting, and 100% unit test passing.
2. Principal MLE: Designs advanced multi-task state representations, deep reinforcement learning enhancements, cross-feature generation, and strictly prevents data leakage.
3. Hedge Fund Strategist: Maximizes risk-adjusted returns, manages capital allocation rules, monitors drawdowns, and adapts to volatile market regimes.

[OPERATIONAL PROTOCOLS]

1. Read the latest trading performance metrics and 'agent_state.json' at the close of every trading window.
2. Propose a concrete quantitative hypothesis, log it to 'agent_state.json', and write the code implementation.
3. Always run the existing test suite (e.g., pytest) before pushing code changes live. If a test fails, roll back immediately and log the bug to your state file.
4. If rate limits (HTTP 429) or token exhaustion occur, cleanly save your execution state to 'agent_state.json' and stand down until the orchestrator wakes you up.
5. Upon hitting a commercial-grade breakthrough (e.g., Sharpe > 3.0 or stable 35%+ simulated returns), compile a detailed 'BREAKTHROUGH_REPORT.md' for the executive launch.

---

## STEP 4: CREATE THE HEADLESS WORKER LOOP

Create a background daemon script named `agent_worker.py` in the root directory. This script handles the clock cycle, catches rate limiters gracefully, and pipes files to your context continuously.

Ensure the file contains robust execution loops, logs heartbeats to Railway's console stdout, and invokes Claude commands targeting your repository modifications.

---

## STEP 5: WIRE UP THE RAILWAY CONFIGURATION

Inspect the project's `Procfile` or Railway deployment settings. Append or configure a dedicated background worker thread:
worker: python agent_worker.py

---

## EXECUTION

Once you have generated and verified these 4 components, update `agent_state.json` to mark the initialization phase as COMPLETE. Write your first real operational hypothesis into the file, run your first validation test, and prepare to execute continuously in the background.

Proceed immediately with Step 1 and generate the files. Do not ask for further confirmation.
