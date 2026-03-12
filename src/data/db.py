"""TimescaleDB connection, schema definition, and initialization.

Tables (all created as TimescaleDB hypertables where applicable):
  ohlcv_1m, ohlcv_5m, ohlcv_1h, ohlcv_1d  — price bars
  options_flow                               — unusual options flow events
  news_raw                                   — raw news articles
  feature_matrix                             — computed FFSA feature rows
  signals                                    — per-ticker model outputs
  trades                                     — execution log
"""

from __future__ import annotations

import logging
import structlog
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import text

from src.config import get_settings

logger = structlog.get_logger(__name__)


# ─── SQLAlchemy base ──────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ─── OHLCV tables (shared schema, separate hypertables per resolution) ─────────

def _ohlcv_columns() -> list[Column]:
    return [
        Column("time", TIMESTAMP(timezone=True), primary_key=True, nullable=False),
        Column("ticker", String(16), primary_key=True, nullable=False),
        Column("open", Float, nullable=False),
        Column("high", Float, nullable=False),
        Column("low", Float, nullable=False),
        Column("close", Float, nullable=False),
        Column("volume", Float, nullable=False),
        Column("vwap", Float),
        Column("transactions", Integer),
    ]


class OHLCV1m(Base):
    __tablename__ = "ohlcv_1m"
    __table_args__ = (
        UniqueConstraint("time", "ticker", name="uq_ohlcv_1m_time_ticker"),
        Index("ix_ohlcv_1m_ticker_time", "ticker", "time"),
    )
    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String(16), primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)
    transactions = Column(Integer)


class OHLCV5m(Base):
    __tablename__ = "ohlcv_5m"
    __table_args__ = (
        UniqueConstraint("time", "ticker", name="uq_ohlcv_5m_time_ticker"),
        Index("ix_ohlcv_5m_ticker_time", "ticker", "time"),
    )
    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String(16), primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)
    transactions = Column(Integer)


class OHLCV1h(Base):
    __tablename__ = "ohlcv_1h"
    __table_args__ = (
        UniqueConstraint("time", "ticker", name="uq_ohlcv_1h_time_ticker"),
        Index("ix_ohlcv_1h_ticker_time", "ticker", "time"),
    )
    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String(16), primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)
    transactions = Column(Integer)


class OHLCV1d(Base):
    __tablename__ = "ohlcv_1d"
    __table_args__ = (
        UniqueConstraint("time", "ticker", name="uq_ohlcv_1d_time_ticker"),
        Index("ix_ohlcv_1d_ticker_time", "ticker", "time"),
    )
    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String(16), primary_key=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)
    transactions = Column(Integer)


# ─── Options Flow ─────────────────────────────────────────────────────────────


class OptionsFlow(Base):
    """Unusual options flow events from Unusual Whales."""

    __tablename__ = "options_flow"
    __table_args__ = (Index("ix_options_flow_ticker_time", "ticker", "time"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP(timezone=True), nullable=False)
    ticker = Column(String(16), nullable=False)
    contract = Column(String(32))          # e.g. "AAPL240119C00150000"
    option_type = Column(String(4))        # "call" or "put"
    strike = Column(Float)
    expiry = Column(TIMESTAMP(timezone=True))
    premium = Column(Float)               # total premium paid ($)
    volume = Column(Integer)
    open_interest = Column(Integer)
    vol_oi_ratio = Column(Float)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    net_gex = Column(Float)               # Gamma Exposure contribution
    smart_money_score = Column(Float)     # premium-weighted directional score
    unusual_flag = Column(Boolean, default=False)
    put_call_ratio = Column(Float)        # total put vol / total call vol
    iv_rank = Column(Float)               # put IV - call IV skew (positive = fear)
    raw = Column(JSONB)                   # original API payload


# ─── News ─────────────────────────────────────────────────────────────────────


class NewsRaw(Base):
    """Raw news articles before sentiment scoring."""

    __tablename__ = "news_raw"
    __table_args__ = (Index("ix_news_raw_ticker_published", "ticker", "published_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
    ticker = Column(String(16), nullable=False)
    headline = Column(Text, nullable=False)
    body = Column(Text)
    source = Column(String(64))
    url = Column(Text)
    sentiment_score = Column(Float)       # filled by FinBERT after scoring
    sentiment_label = Column(String(16))  # "positive" / "negative" / "neutral"
    relevance_score = Column(Float)       # how specifically about this ticker
    raw = Column(JSONB)


# ─── Feature Matrix ────────────────────────────────────────────────────────────


class FeatureMatrix(Base):
    """Computed FFSA feature rows aligned to 1m bar timestamps."""

    __tablename__ = "feature_matrix"
    __table_args__ = (
        UniqueConstraint("time", "ticker", name="uq_feature_matrix_time_ticker"),
        Index("ix_feature_matrix_ticker_time", "ticker", "time"),
    )

    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String(16), primary_key=True, nullable=False)
    features = Column(JSONB, nullable=False)   # dict of feature_name → float
    ffsa_version = Column(String(16))          # which FFSA run produced this


# ─── Signals ─────────────────────────────────────────────────────────────────


class Signal(Base):
    """Per-ticker model output at each bar."""

    __tablename__ = "signals"
    __table_args__ = (
        UniqueConstraint("time", "ticker", name="uq_signals_time_ticker"),
        Index("ix_signals_ticker_time", "ticker", "time"),
    )

    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    ticker = Column(String(16), primary_key=True, nullable=False)
    transformer_direction = Column(Float)   # +1 long, 0 flat, -1 short
    transformer_confidence = Column(Float)
    tcn_direction = Column(Float)
    tcn_confidence = Column(Float)
    sentiment_index = Column(Float)
    ensemble_signal = Column(Float)        # weighted combination
    rl_action = Column(String(32))         # hold/buy_small/buy_medium/...
    rl_q_value = Column(Float)


# ─── Trades ───────────────────────────────────────────────────────────────────


class Trade(Base):
    """Full execution log with signal attribution."""

    __tablename__ = "trades"
    __table_args__ = (Index("ix_trades_ticker_entry", "ticker", "entry_time"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(8), nullable=False)   # "paper" or "live"
    ticker = Column(String(16), nullable=False)
    side = Column(String(8), nullable=False)   # "buy" or "sell"
    entry_time = Column(TIMESTAMP(timezone=True), nullable=False)
    exit_time = Column(TIMESTAMP(timezone=True))
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    shares = Column(Float, nullable=False)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String(64))           # trailing_stop / signal_reversal / ...
    # Signal state at entry
    transformer_direction = Column(Float)
    transformer_confidence = Column(Float)
    tcn_direction = Column(Float)
    tcn_confidence = Column(Float)
    sentiment_index = Column(Float)
    ensemble_signal = Column(Float)
    rl_action = Column(String(32))
    rl_q_value = Column(Float)
    alpaca_order_id = Column(String(64))


# ─── Engine & session factory ──────────────────────────────────────────────────

_engine = None
_async_session: async_sessionmaker | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            echo=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session
    if _async_session is None:
        _async_session = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _async_session


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an AsyncSession."""
    async with get_session_factory()() as session:
        yield session


# ─── Schema initialization ────────────────────────────────────────────────────

_HYPERTABLES = [
    ("ohlcv_1m", "time"),
    ("ohlcv_5m", "time"),
    ("ohlcv_1h", "time"),
    ("ohlcv_1d", "time"),
    ("options_flow", "time"),
    ("news_raw", "published_at"),
    ("feature_matrix", "time"),
    ("signals", "time"),
]

_COMPRESSION_POLICIES = [
    # Compress chunks older than 7 days (saves ~90% storage)
    ("ohlcv_1m", "7 days"),
    ("ohlcv_5m", "7 days"),
    ("feature_matrix", "7 days"),
]


async def init_db() -> None:
    """Create all tables and convert to TimescaleDB hypertables (idempotent).

    Falls back gracefully to plain PostgreSQL when TimescaleDB is not installed
    (e.g. local dev without Docker). Tables are still created; hypertable
    conversion is skipped with a warning.
    """
    engine = get_engine()

    # ── Step 1: Try to enable TimescaleDB (non-fatal if not installed) ────────
    timescaledb_available = False
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        timescaledb_available = True
        logger.info("timescaledb_extension_ready")
    except Exception as exc:
        logger.warning(
            "timescaledb_unavailable — running with plain PostgreSQL (hypertables skipped): %s",
            exc,
        )

    # ── Step 2: Create all ORM tables (always runs) ───────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ── Step 2b: Additive column migrations (idempotent, IF NOT EXISTS) ───────
    _migrations = [
        "ALTER TABLE options_flow ADD COLUMN IF NOT EXISTS put_call_ratio FLOAT",
        "ALTER TABLE options_flow ADD COLUMN IF NOT EXISTS iv_rank FLOAT",
    ]
    async with engine.begin() as conn:
        for stmt in _migrations:
            try:
                await conn.execute(text(stmt))
            except Exception as exc:
                logger.warning("migration_skipped: %s — %s", stmt, exc)

    # ── Step 3: Convert to hypertables (skipped if TimescaleDB unavailable) ─────
    if not timescaledb_available:
        logger.info("db_initialized (plain_postgresql — no hypertables)")
        return

    async with engine.connect() as conn:
        for table, time_col in _HYPERTABLES:
            try:
                await conn.execute(
                    text(
                        f"SELECT create_hypertable('{table}', '{time_col}', "
                        f"if_not_exists => TRUE, migrate_data => TRUE);"
                    )
                )
                await conn.commit()
                logger.info("hypertable_ready: %s (time_col=%s)", table, time_col)
            except Exception as exc:
                await conn.rollback()
                logger.warning("hypertable_skip %s: %s", table, exc)

        # Enable compression on large tables
        for table, older_than in _COMPRESSION_POLICIES:
            try:
                await conn.execute(
                    text(f"ALTER TABLE {table} SET (timescaledb.compress);")
                )
                await conn.execute(
                    text(
                        f"SELECT add_compression_policy('{table}', "
                        f"INTERVAL '{older_than}', if_not_exists => TRUE);"
                    )
                )
                await conn.commit()
            except Exception as exc:
                await conn.rollback()
                logger.warning("compression_skip %s: %s", table, exc)

    logger.info("db_initialized")
