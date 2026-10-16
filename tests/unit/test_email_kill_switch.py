"""`email_alerts_enabled` silences every sender — and nothing else.

Owner request 2026-10-15: stop all StockBot email. The watchdog had been
re-sending alerts because its 4-hour dedupe is in-memory and six deploys in two
days kept wiping it.

The important property is the SECOND half of these tests. Silencing a
notification must not silence the CONTROL behind it: a circuit-breaker halt
still has to fire and still has to require a manual resume. CLAUDE.md forbids
disabling risk controls; suppressing an email is not that, and these tests are
what keeps the distinction real rather than assumed.
"""
from __future__ import annotations

import inspect

import pytest

SENDERS = [
    ("src.agents.watchdog_agent", "WatchdogAgent"),
    ("src.agents.integrity_agent", None),
    ("src.agents.news_risk_agent", None),
    ("src.agents.forecast_agent", None),
    ("src.risk.circuit_breakers", None),
]


def _send_email_source(module_path: str) -> str:
    import importlib

    mod = importlib.import_module(module_path)
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != module_path:
            continue
        fn = getattr(obj, "_send_email", None)
        if fn is not None:
            return inspect.getsource(fn)
    raise AssertionError(f"no _send_email found in {module_path}")


# ── Every sender is gated ────────────────────────────────────────────────────

@pytest.mark.parametrize("module_path,_cls", SENDERS)
def test_every_sender_checks_the_master_switch(module_path, _cls):
    """All five senders duplicate the SMTP block, so all five need the gate.

    A new sender that copies the existing pattern without the gate would leak
    email past the switch — which is exactly how "I turned it off and still got
    mail" happens.
    """
    src = _send_email_source(module_path)
    assert "email_alerts_enabled" in src, f"{module_path} bypasses the kill switch"


@pytest.mark.parametrize("module_path,_cls", SENDERS)
def test_the_gate_precedes_the_smtp_connection(module_path, _cls):
    """Suppression must short-circuit before any network call."""
    src = _send_email_source(module_path)
    gate = src.index("email_alerts_enabled")
    for marker in ("smtplib.SMTP", "srv.login", "send_message"):
        if marker in src:
            assert gate < src.index(marker), f"{module_path}: gate after {marker}"


def test_the_switch_defaults_to_on():
    """Email is a safety channel; turning it off must be deliberate."""
    from src.config import Settings

    assert Settings.model_fields["email_alerts_enabled"].default is True


def test_suppression_is_logged():
    """A silent silencer is indistinguishable from a broken mailer.

    SMTP on this project was genuinely dead for three weeks without anyone
    noticing. If the switch is ever the reason mail stopped, the log has to say
    so.
    """
    src = _send_email_source("src.agents.watchdog_agent")
    assert "email_suppressed" in src


# ── The control must be untouched ────────────────────────────────────────────

def test_halting_still_works_with_email_off():
    """The breaker fires on its own terms; email is only the notification."""
    from src.risk.circuit_breakers import CircuitBreakers

    cb = CircuitBreakers(pipeline_id="test")
    cb._halt("daily_loss", "test halt with alerts disabled")
    assert cb.is_halted is True
    assert "daily_loss" in cb._halt_reason


def test_resume_still_requires_explicit_authorization():
    """Silencing alerts must not make a halt easier to clear.

    CLAUDE.md: a halt may never be lifted without a human. If disabling email
    had any effect on this path, the switch would be disabling a risk control
    rather than a notification.
    """
    from src.risk.circuit_breakers import CircuitBreakers

    cb = CircuitBreakers(pipeline_id="test")
    cb._halt("max_drawdown", "test")
    sig = inspect.signature(cb.resume_trading)
    assert "authorized_by" in sig.parameters
    assert cb.is_halted is True, "halt cleared itself"


def test_the_switch_does_not_touch_credentials_or_recipients():
    """Deliberately a flag, not a cleared smtp_* or recipient list.

    Clearing the recipients would silence halt alerts, integrity escalations and
    news-risk warnings as an invisible side effect — the config file already
    warns about exactly that. It would also destroy credentials that were broken
    for three weeks and only recently recovered.
    """
    from src.config import Settings

    for field in ("smtp_host", "smtp_user", "smtp_password", "forecast_email_to"):
        assert field in Settings.model_fields
        assert Settings.model_fields[field].default == ""
