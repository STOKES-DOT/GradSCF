"""Compatibility import; implementation lives in gradscf.integrals.basis."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("gradscf.integrals.basis")
