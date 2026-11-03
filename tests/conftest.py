"""Shared test fixtures. FakeLLM is added by Task 7."""

from pathlib import Path

import pytest

from make_fixtures import build_fixtures


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return build_fixtures(Path(__file__).parent / "fixtures")
