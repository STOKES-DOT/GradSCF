"""Compatibility module for shared AO-pair layouts."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module("gradscf.integrals.layouts")
