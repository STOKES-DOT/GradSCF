"""Preserve eager finite-data validation across the cached JIT boundary."""
from contextlib import contextmanager
from contextvars import ContextVar

import jax.numpy as jnp

_CHECKS = ContextVar('td_response_finite_checks', default=None)


@contextmanager
def finite_checks():
    checks = []
    token = _CHECKS.set(checks)
    try:
        yield checks
    finally:
        _CHECKS.reset(token)


def record_finite_check(value):
    checks = _CHECKS.get()
    if checks is not None:
        checks.append(jnp.all(jnp.isfinite(value)))
