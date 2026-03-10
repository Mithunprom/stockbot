"""StockBot — FastAPI entry point.

Routes:
  GET  /health            → liveness check
  GET  /signals           → latest ensemble signals for trading universe
  GET  /trades            → recent trade history with full attribution
  GET  /reports/{name}    → latest named report (risk, latency, drift, ...)
  WS   /ws/dashboard      → real-time push: signals, PnL, risk metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.data.alpaca_ws import AlpacaDataStreamClient
from src.data.db import get_session_factory, init_db
from src.data.news import NewsPoller
from src.data.options_flow import OptionsFlowPoller

logger = structlog.get_logger(__name__)

# Default trading universe — overridden by config/paper.yaml or config/live.yaml
_DEFAULT_UNIVERSE: list[str] = [
    # Core mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    # Semiconductors
    "AVGO", "AMD", "SNDK",
    # Defense / military
    "LMT", "RTX", "NOC", "GD", "BA",
    # Financials
    "JPM", "V", "MA",
    # AI / high-momentum
    "PLTR", "ARM", "MSTR",
    # Energy
    "XOM", "CVX",
    # Consumer / healthcare
    "LLY", "UNH", "COST", "NFLX",
]

# ─── Module-level singletons (populated in lifespan) ──────────────────────────

_signal_loop: Any | None = None    # SignalLoop (imported lazily to avoid circular)


# ─── Startup / Shutdown ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global _signal_loop

    settings = get_settings()
    logger.info(
        "stockbot_starting",
        mode=settings.alpaca_mode,
        environment=settings.environment,
    )

    # Initialize TimescaleDB schema (idempotent)
    await init_db()
    session_factory = get_session_factory()

    # ── Trading universe (screener → config → defaults) ───────────────────────
    universe: list[str] = _load_universe()

    # ── Phase 1: Real-time data ingest ────────────────────────────────────────
    alpaca_stream = AlpacaDataStreamClient(tickers=universe, feed="iex")

    # ── Phase 3+: Incremental feature computation on every live bar ───────────
    # Must be registered before stream starts so no bars are missed.
    # Feature cols loaded after this block — register the callback lazily via closure.
    from src.features.live import LiveFeatureComputer
    _live_feature_computer: LiveFeatureComputer | None = None

    async def _on_1m_bar(ticker: str, bar_time: Any) -> None:
        if _live_feature_computer is not None:
            await _live_feature_computer.on_bar(ticker, bar_time)

    alpaca_stream.on_1m_bar = _on_1m_bar

    stream_task = asyncio.create_task(
        alpaca_stream.start(), name="alpaca_data_stream"
    )
    logger.info("alpaca_stream_started", tickers=len(universe), feed="iex")

    options_poller = OptionsFlowPoller(universe=universe, poll_interval_seconds=300)
    options_task = asyncio.create_task(
        options_poller.start(), name="options_flow_poller"
    )

    news_poller = NewsPoller(universe=set(universe))
    news_task = asyncio.create_task(news_poller.start(), name="news_poller")

    # ── Phase 3: Load FFSA feature list ──────────────────────────────────────
    feature_cols: list[str] = _load_ffsa_features()
    logger.info("ffsa_features_loaded", n_features=len(feature_cols))

    # Activate live feature computer now that feature_cols are known
    _live_feature_computer = LiveFeatureComputer(feature_cols)
    logger.info("live_feature_computer_ready")

    # ── Download ML checkpoints from S3 (if not already present) ────────────
    await _download_models_from_s3()

    # ── Phase 3+5: Load ML models ─────────────────────────────────────────────
    from src.models.ensemble import EnsembleEngine, EnsembleWeights

    ensemble = EnsembleEngine()
    try:
        await ensemble.load()
        logger.info("ensemble_models_loaded")
    except Exception as exc:
        logger.warning("ensemble_load_partial", error=str(exc))

    # ── Phase 5: Execution stack ──────────────────────────────────────────────
    from src.execution.alpaca import AlpacaOrderRouter
    from src.execution.position_manager import PositionManager
    from src.risk.circuit_breakers import CircuitBreakers

    circuit_breakers = CircuitBreakers()

    # Sync initial portfolio value from broker
    alpaca_router = AlpacaOrderRouter(circuit_breakers)
    initial_portfolio = 100_000.0
    try:
        account = await alpaca_router.get_account()
        initial_portfolio = account.get("portfolio_value", 100_000.0)
        logger.info("portfolio_synced", value=initial_portfolio)
    except Exception as exc:
        logger.warning("portfolio_sync_failed_using_default", error=str(exc))

    pos_manager = PositionManager(initial_portfolio=initial_portfolio)

    # ── Phase 5: Signal loop ──────────────────────────────────────────────────
    from src.agents.signal_loop import SignalLoop

    _signal_loop = SignalLoop(
        universe=universe,
        ensemble=ensemble,
        alpaca=alpaca_router,
        circuit_breakers=circuit_breakers,
        pos_manager=pos_manager,
        session_factory=session_factory,
        feature_cols=feature_cols,
        broadcast_fn=broadcast_dashboard,
    )
    signal_task = asyncio.create_task(
        _signal_loop.start(), name="signal_loop"
    )
    logger.info("signal_loop_started")

    # ── Phase 6: Sub-agent scheduler ─────────────────────────────────────────
    from src.agents.latency_agent import LatencyAgent
    from src.agents.profit_agent import ProfitAgent
    from src.agents.risk_agent import RiskAgent
    from src.agents.screener_agent import ScreenerAgent
    from src.agents.scheduler import create_scheduler

    risk_agent = RiskAgent(circuit_breakers, pos_manager, alpaca_router)
    latency_agent = LatencyAgent()
    profit_agent = ProfitAgent()
    screener_agent = ScreenerAgent(signal_loop=_signal_loop)

    scheduler = create_scheduler(
        risk_agent=risk_agent,
        latency_agent=latency_agent,
        profit_agent=profit_agent,
        screener_agent=screener_agent,
        mode=settings.alpaca_mode,
    )
    scheduler.start()
    logger.info("scheduler_started", jobs=len(scheduler.get_jobs()))

    # ── Immediate first-run for all sub-agents (populate reports at startup) ──
    # Scheduled jobs only fire at their next cron slot (e.g. 4:30pm).
    # By running each agent once now, reports/risk/live.json and
    # reports/latency/YYYY-MM-DD.json are populated right away so the
    # /status endpoint returns real data from the very first request.
    async def _run_startup_agents() -> None:
        # Small delay to let broker connection settle
        await asyncio.sleep(10)
        logger.info("startup_agents_running")
        # Run screener first so universe is fresh from the very first tick
        try:
            await screener_agent.run()
            logger.info("startup_screener_agent_done")
        except Exception as exc:
            logger.warning("startup_screener_failed", error=str(exc))
        try:
            await risk_agent.run()
            logger.info("startup_risk_agent_done")
        except Exception as exc:
            logger.warning("startup_risk_agent_failed", error=str(exc))
        try:
            await latency_agent.run()
            logger.info("startup_latency_agent_done")
        except Exception as exc:
            logger.warning("startup_latency_agent_failed", error=str(exc))
        if profit_agent is not None:
            try:
                await profit_agent.run(mode=settings.alpaca_mode)
                logger.info("startup_profit_agent_done")
            except Exception as exc:
                logger.warning("startup_profit_agent_failed", error=str(exc))

    asyncio.create_task(_run_startup_agents(), name="startup_agents")
    logger.info("startup_agents_task_created")

    logger.info("stockbot_ready", universe_size=len(universe))
    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    logger.info("stockbot_shutting_down")
    scheduler.shutdown(wait=False)
    if _signal_loop:
        await _signal_loop.stop()
    await alpaca_stream.stop()
    await options_poller.stop()
    await news_poller.stop()
    for task in (stream_task, options_task, news_task, signal_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ─── FFSA feature loader ──────────────────────────────────────────────────────


async def _download_models_from_s3() -> None:
    """Download Transformer and TCN checkpoints from S3 at startup.

    Only downloads if the file doesn't already exist locally (idempotent).
    Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET env vars.
    Silently skips if boto3 is not installed or credentials are missing.
    """
    import os

    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        logger.info("s3_download_skipped_no_bucket")
        return

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        logger.warning("s3_download_skipped_boto3_not_installed")
        return

    # Map S3 key → local path
    model_files = {
        "transformer/step_055314_sharpe_0.896.pt": Path("models/transformer/step_055314_sharpe_0.896.pt"),
        "tcn/step_043461_sharpe_0.776.pt": Path("models/tcn/step_043461_sharpe_0.776.pt"),
        "rl_agent/ppo_500000_steps.zip": Path("models/rl_agent/periodic/ppo_500000_steps.zip"),
    }

    loop = asyncio.get_event_loop()

    def _download() -> None:
        s3 = boto3.client("s3")
        for s3_key, local_path in model_files.items():
            if local_path.exists():
                logger.info("s3_model_already_present", path=str(local_path))
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                logger.info("s3_downloading", bucket=bucket, key=s3_key)
                s3.download_file(bucket, s3_key, str(local_path))
                logger.info("s3_download_complete", path=str(local_path), size=local_path.stat().st_size)
            except (BotoCoreError, ClientError) as exc:
                logger.error("s3_download_failed", key=s3_key, error=str(exc))

    await loop.run_in_executor(None, _download)


def _load_universe() -> list[str]:
    """Load trading universe.

    Priority:
      1. config/universe.json  — written nightly by ScreenerAgent
      2. _DEFAULT_UNIVERSE     — hardcoded fallback (first run / no screener yet)
    """
    universe_file = Path("config/universe.json")
    if universe_file.exists():
        with open(universe_file) as f:
            data = json.load(f)
        symbols = data.get("symbols", [])
        if symbols:
            logger.info(
                "universe_loaded_from_screener",
                count=len(symbols),
                updated_at=data.get("updated_at", "unknown"),
            )
            return symbols
    logger.info("universe_using_default", count=len(_DEFAULT_UNIVERSE))
    return _DEFAULT_UNIVERSE


def _load_ffsa_features() -> list[str]:
    """Load top FFSA feature names from the most recent drift report.

    Search order:
      1. reports/drift/ffsa_*.json  (runtime-generated, most up-to-date)
      2. config/ffsa_features.json  (committed fallback for Railway / fresh deploys)
    """
    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if ffsa_files:
        source = ffsa_files[0]
        logger.info("ffsa_report_found", file=str(source))
    else:
        # Fallback: committed copy shipped with the repo
        source = Path("config/ffsa_features.json")
        if source.exists():
            logger.info("ffsa_report_using_committed_fallback", file=str(source))
        else:
            logger.warning("ffsa_report_not_found_using_empty_list")
            return []

    with open(source) as f:
        data = json.load(f)
    features = data.get("selected_features", [])
    logger.info("ffsa_features_loaded_detail", n=len(features), source=str(source))
    return features


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StockBot API",
    description="Autonomous trading bot — ML signal ensemble + RL execution",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Routes ──────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> JSONResponse:
    """Root route — lists available API endpoints."""
    return JSONResponse(content={
        "name": "StockBot API",
        "version": "0.1.0",
        "mode": get_settings().alpaca_mode,
        "endpoints": {
            "health":  "GET /health",
            "signals": "GET /signals",
            "trades":  "GET /trades",
            "reports": "GET /reports/{risk|latency|drift|opportunities}",
            "dashboard_ws": "WS /ws/dashboard",
        },
        "ui": "Run 'cd frontend && npm run dev' then open http://localhost:5173",
    })


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "mode": get_settings().alpaca_mode,
        "signal_loop_active": _signal_loop is not None,
    }


@app.get("/signals")
async def get_signals(limit: int = 50) -> JSONResponse:
    """Return the latest ensemble signals for each ticker in the universe."""
    if _signal_loop is None:
        return JSONResponse(content={
            "signals": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Signal loop not yet initialized",
        })

    signals = _signal_loop.get_latest_signals()[:limit]
    return JSONResponse(content={
        "signals": signals,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(signals),
    })


@app.get("/trades")
async def get_trades(limit: int = 100, mode: str = "paper") -> JSONResponse:
    """Return recent trade history with full signal attribution."""
    from sqlalchemy import select

    from src.data.db import Trade, get_session_factory

    try:
        sf = get_session_factory()
        async with sf() as session:
            rows = await session.execute(
                select(Trade)
                .where(Trade.mode == mode)
                .order_by(Trade.entry_time.desc())
                .limit(limit)
            )
            trades = rows.scalars().all()
            trade_list = [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "side": t.side,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "ensemble_signal": t.ensemble_signal,
                    "transformer_confidence": t.transformer_confidence,
                    "tcn_confidence": t.tcn_confidence,
                    "sentiment_index": t.sentiment_index,
                }
                for t in trades
            ]
    except Exception as exc:
        logger.warning("trades_query_failed", error=str(exc))
        trade_list = []

    return JSONResponse(content={
        "trades": trade_list,
        "mode": mode,
        "count": len(trade_list),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/status")
async def get_status() -> JSONResponse:
    """Comprehensive system status — models, agents, weights, RL, circuit breakers."""
    import pathlib

    status: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": get_settings().alpaca_mode,
    }

    # ── Signal loop ───────────────────────────────────────────────────────────
    if _signal_loop is not None:
        status["signal_loop"] = {
            "active": True,
            "universe_size": len(_signal_loop._universe),
            "universe": _signal_loop._universe,
            "feature_cols": len(_signal_loop._feature_cols),
            "consecutive_losses": _signal_loop._consecutive_losses,
            "open_positions": len(_signal_loop._pm._positions),
            "portfolio_value": round(_signal_loop._pm.portfolio_value, 2),
            "portfolio_heat": round(_signal_loop._pm.portfolio_heat, 4),
            "daily_start_value": round(_signal_loop._daily_start_value, 2),
            "daily_pnl_pct": round(
                (_signal_loop._pm.portfolio_value / max(_signal_loop._daily_start_value, 1) - 1) * 100, 3
            ),
        }

        # ── Ensemble weights & model status ───────────────────────────────────
        ens = _signal_loop._ensemble
        status["ensemble"] = {
            "weights": {
                "transformer": ens.weights.transformer,
                "tcn": ens.weights.tcn,
                "sentiment": ens.weights.sentiment,
            },
            "transformer_loaded": ens._transformer is not None,
            "tcn_loaded": ens._tcn is not None,
            "sentiment_loaded": ens._sentiment is not None,
            "finbert_active": bool(os.environ.get("HUGGINGFACE_API_TOKEN")),
            "sentiment_note": (
                "FinBERT active via HF Inference API"
                if os.environ.get("HUGGINGFACE_API_TOKEN")
                else "Sentiment disabled — set HUGGINGFACE_API_TOKEN"
            ),
        }

        # ── Circuit breakers ──────────────────────────────────────────────────
        cb = _signal_loop._cb
        status["circuit_breakers"] = {
            "halted": cb.is_halted,
            "halt_reason": cb.halt_reason,
        }

        # ── RL agent ──────────────────────────────────────────────────────────
        rl = _signal_loop._rl_agent
        status["rl_agent"] = {"loaded": rl is not None}
        if rl is not None:
            status["rl_agent"]["policy"] = type(rl.policy).__name__
    else:
        status["signal_loop"] = {"active": False}

    # ── Model checkpoints (read filenames for Sharpe scores) ──────────────────
    def _best_checkpoint(glob_pattern: str) -> dict[str, Any]:
        files = sorted(pathlib.Path(".").glob(glob_pattern), reverse=True)
        if not files:
            return {"found": False}
        best = files[0]
        parts = best.stem.split("sharpe_")
        sharpe = float(parts[1]) if len(parts) > 1 else None
        step_parts = best.stem.split("step_")
        step = int(step_parts[1].split("_")[0]) if len(step_parts) > 1 else None
        return {"found": True, "file": best.name, "sharpe": sharpe, "step": step}

    status["model_checkpoints"] = {
        "transformer": _best_checkpoint("models/transformer/step_*_sharpe_*.pt"),
        "tcn":         _best_checkpoint("models/tcn/step_*_sharpe_*.pt"),
        "rl_agent":    _best_checkpoint("models/rl_agent/best_ppo_*.zip"),
    }

    # ── FFSA feature selection ────────────────────────────────────────────────
    ffsa_files = sorted(pathlib.Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    _ffsa_fallback = pathlib.Path("config/ffsa_features.json")
    if ffsa_files:
        ffsa_source = ffsa_files[0]
    elif _ffsa_fallback.exists():
        ffsa_source = _ffsa_fallback
    else:
        ffsa_source = None

    if ffsa_source:
        with open(ffsa_source) as f:
            ffsa_data = json.load(f)
        status["ffsa"] = {
            "report": ffsa_source.name,
            "source": "drift_report" if ffsa_files else "committed_fallback",
            "n_features": len(ffsa_data.get("selected_features", [])),
            "validation_ic": ffsa_data.get("validation_ic", ffsa_data.get("ic", None)),
            "features": ffsa_data.get("selected_features", []),
        }
    else:
        status["ffsa"] = {"report": None, "note": "No FFSA report — run scripts/run_ffsa.py"}

    # ── Sub-agent last reports ────────────────────────────────────────────────
    def _latest_report(folder: str) -> dict[str, Any]:
        files = sorted(pathlib.Path(f"reports/{folder}").glob("*.json"), reverse=True)
        if not files:
            return {"last_run": None, "file": None}
        with open(files[0]) as f:
            data = json.load(f)
        return {"last_run": files[0].name, "data": data}

    # Screener last run
    screener_files = sorted(
        pathlib.Path("reports/opportunities").glob("screener_*.json"), reverse=True
    )
    screener_status: dict[str, Any] = {"last_run": None}
    if screener_files:
        with open(screener_files[0]) as f:
            sc = json.load(f)
        screener_status = {
            "last_run": screener_files[0].name,
            "date": sc.get("date"),
            "universe_size": len(sc.get("universe", [])),
            "top_5_momentum": sc.get("top_50_by_score", [])[:5],
        }

    status["sub_agents"] = {
        "risk_agent":    _latest_report("risk"),
        "latency_agent": _latest_report("latency"),
        "profit_agent":  _latest_report("opportunities"),
        "screener_agent": screener_status,
    }

    # ── Staging proposals (Profit Agent suggestions) ──────────────────────────
    staging = pathlib.Path("config/staging/profit_suggestions.json")
    if staging.exists():
        with open(staging) as f:
            status["staging_proposals"] = json.load(f)
    else:
        status["staging_proposals"] = None

    return JSONResponse(content=status)


@app.get("/reports/{report_name}")
async def get_report(report_name: str) -> JSONResponse:
    """Return the latest JSON report by name (risk, latency, drift, opportunities)."""
    import pathlib

    valid = {"risk", "latency", "drift", "opportunities"}
    if report_name not in valid:
        return JSONResponse(
            status_code=404, content={"error": f"Unknown report: {report_name}"}
        )

    report_dir = pathlib.Path(f"reports/{report_name}")
    files = sorted(report_dir.glob("*.json"), reverse=True)
    if not files:
        return JSONResponse(
            content={"report": report_name, "data": None, "note": "No reports yet"}
        )

    with open(files[0]) as f:
        data = json.load(f)
    return JSONResponse(
        content={"report": report_name, "file": files[0].name, "data": data}
    )


# ─── WebSocket Dashboard ──────────────────────────────────────────────────────

_ws_clients: list[WebSocket] = []


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    """Push live signals, PnL, and risk metrics to the React dashboard."""
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("dashboard_ws_connected", clients=len(_ws_clients))

    try:
        while True:
            # Send heartbeat; real pushes come from signal_loop via broadcast_dashboard
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        logger.info("dashboard_ws_disconnected", clients=len(_ws_clients))


async def broadcast_dashboard(payload: dict[str, Any]) -> None:
    """Broadcast a payload to all connected dashboard WebSocket clients."""
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
