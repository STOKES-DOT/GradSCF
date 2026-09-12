"""Compatibility import; implementation lives in gradscf.integrals.backends.jax_reference.direct_jk."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("gradscf.integrals.backends.jax_reference.direct_jk")
