"""Isolation shared by every test."""
import pytest

import models


@pytest.fixture(autouse=True)
def _no_leftover_cooldowns():
    """Provider cooldowns are process-wide and clock-based. A test that drives
    a provider into one must not decide which provider the next test reaches."""
    models._nvidia_cooldown.clear()
    yield
    models._nvidia_cooldown.clear()
