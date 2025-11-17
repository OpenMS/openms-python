"""Test configuration helpers for environments without pytest-cov."""

try:  # pragma: no cover - best-effort detection
    import pytest_cov  # type: ignore
except ImportError:  # pragma: no cover - executed when plugin missing
    def pytest_addoption(parser):
        parser.addoption("--cov", action="append", default=[], help="No-op without pytest-cov")
        parser.addoption("--cov-report", action="append", default=[], help="No-op without pytest-cov")
