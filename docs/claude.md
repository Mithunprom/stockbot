# Project: Autonomous Stockbot v1.0

## Objective
Build a system that identifies the Top 20 stocks via Alpha factors, trains an RL agent for entry/exit signals, and executes paper trades before going live.

## Technical Architecture


### 1. Data Pipeline (`data_engine.py`)
* Source: YFinance (History) & Alpaca/Polygon (Real-time).
* Features: RSI, MACD, Bollinger Bands, Volume Profile, and Sentiment Analysis.
* Output: Top 20 stocks based on 14-day momentum and liquidity.

### 2. The RL Environment (`trading_env.py`)
* State: [Balance, Positions, 5-day Indicator History, Unrealized PnL].
* Action Space: Discrete (0: Sell All, 1: Hold, 2: Buy Max) or Continuous (Weighting).
* Auto-Switching: Logic to compare PPO vs. A2C performance and hot-swap the model.

### 3. Paper Trading & Tracking (`tracker.py`)
* Simulated Execution: Mimics real-market latency.
* Log Gain/Loss: Daily CSV/JSON exports of the equity curve.
* Stability Check: If Max Drawdown > 15%, the bot must auto-stop and revert to "Observation Mode."

### 4. Live Execution Module (`broker_bridge.py`)
* Target API: Alpaca Markets (Commission-free).
* Requirement: Multi-factor authentication and secure key handling (Environment Variables).

## Milestones
- [ ] Phase 1a: Historical Data & Technical Indicator Pipeline.
- [ ] Phase 1b: Builing all the technical analysis and models.
- [ ] Phase 2: Gymnasium Environment Setup & Reward Function Tuning.
- [ ] Phase 3a: Integration of RL Ensemble (Switching Algos).
- [ ] Phase 3b: Make the algorithm works such that Profit is maximized. 
- [ ] Phase 4: Real-time Paper Trading WebSocket implementation.