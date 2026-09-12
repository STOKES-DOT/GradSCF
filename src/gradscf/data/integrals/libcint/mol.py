"""Compatibility import; implementation lives in gradscf.integrals.backends.pyscf_mol."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("gradscf.integrals.backends.pyscf_mol")
