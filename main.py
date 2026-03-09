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
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "ORCL", "CRM", "PLTR", "COIN", "JPM", "V", "MA",
    "LLY", "UNH", "XOM", "CVX", "HD", "COST", "NFLX", "DIS",
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

    # ── Trading universe (load from config or use defaults) ───────────────────
    cfg = settings.get_trading_config()
    universe: list[str] = cfg.get("universe", {}).get("symbols", _DEFAULT_UNIVERSE)

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
    from src.agents.scheduler import create_scheduler

    risk_agent = RiskAgent(circuit_breakers, pos_manager, alpaca_router)
    latency_agent = LatencyAgent()
    profit_agent = ProfitAgent()

    scheduler = create_scheduler(
        risk_agent=risk_agent,
        latency_agent=latency_agent,
        profit_agent=profit_agent,   # enabled from Day 1 in paper mode
        mode=settings.alpaca_mode,
    )
    scheduler.start()
    logger.info("scheduler_started", jobs=len(scheduler.get_jobs()))

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


def _load_ffsa_features() -> list[str]:
    """Load top FFSA feature names from the most recent drift report."""
    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if not ffsa_files:
        logger.warning("ffsa_report_not_found_using_empty_list")
        return []
    with open(ffsa_files[0]) as f:
        data = json.load(f)
    return data.get("selected_features", [])


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
