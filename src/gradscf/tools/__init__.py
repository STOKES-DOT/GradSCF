"""Utility modules for GradSCF (``gradscf.tools``).

Hosts the former top-level helper modules:

- ``api``: simplified strict-JAX public API entry points
- ``device``: JAX device placement helpers
- ``features``: grid-based XC feature computation
- ``jax_runtime``: JAX runtime configuration (persistent compile cache)
- ``protocols``: typing Protocols for XC functionals and models
- ``spectra``: unit conversion and spectral post-processing
- ``types``: shared dataclasses
- ``upstreams``: optional-dependency probing with explicit errors
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = (
    "api",
    "device",
    "features",
    "jax_runtime",
    "protocols",
    "spectra",
    "types",
    "upstreams",
)


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_SUBMODULES)
