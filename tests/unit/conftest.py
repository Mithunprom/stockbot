"""Shared test fixtures for unit tests.

Sets required environment variables so pydantic-settings can instantiate
`Settings` without real API keys. Applies to the entire tests/unit suite.
"""
import os
import pytest


@pytest.fixture(autouse=True)
def _mock_env_vars(monkeypatch):
    """Inject stub values for all required Settings fields."""
    monkeypatch.setenv("POLYGON_API_KEY", "test_polygon_key")
    monkeypatch.setenv("ALPACA_API_KEY", "test_alpaca_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_alpaca_secret")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
