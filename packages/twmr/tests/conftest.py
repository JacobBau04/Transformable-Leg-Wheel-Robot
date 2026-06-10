"""Shared pytest config for the twmr test suite.

Pure-function tests run on CPU and need no GPU/warp. Tests that build the
mujoco_playground env are marked `env`; they are heavier and require the warp
backend. Run only the fast tests with:  pytest -m "not env"
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "env: test builds the mujoco_playground env (needs warp/GPU)"
    )
