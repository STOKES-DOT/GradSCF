"""Narrow compatibility for upstream parameter bindings, without XC formulas."""
from functools import lru_cache, update_wrapper
from types import FunctionType

import numpy as np


@lru_cache(maxsize=16)
def pw_parameter_factory(factory):
    """Preserve PW's three parameter sets across legacy ``get_p`` conversion.

    The jax_xc 0.0.9 converter observed in validation repeats element zero of
    each PW vector. The factory itself supplies all 22 correct scalar values.
    Reconstruct vectors from those actual arguments, including user overrides,
    while retaining the upstream generated formula and its derivative graph.
    A private globals dictionary avoids monkey-patching the installed module.
    """
    if not isinstance(factory, FunctionType) or "get_p" not in factory.__globals__:
        # Factories that do not use the legacy converter have no such boundary.
        return factory
    upstream_get_p = factory.__globals__["get_p"]
    fields = ("pp", "a", "alpha1", "beta1", "beta2", "beta3", "beta4")

    def get_p(name, polarized, *values):
        p = upstream_get_p(name, polarized, *values)
        if name != "lda_c_pw":
            return p
        if len(values) != 22 or not hasattr(p, "params"):
            raise RuntimeError("Unsupported upstream lda_c_pw parameter layout")
        if not all(hasattr(p.params, field) for field in (*fields, "fz20")):
            raise RuntimeError("Unsupported upstream lda_c_pw parameter fields")
        arrays = {field: np.asarray(values[3*i:3*i+3], dtype=np.float64)
                  for i, field in enumerate(fields)}
        return p._replace(params=p.params._replace(**arrays, fz20=np.asarray(values[21], dtype=np.float64)))

    namespace = dict(factory.__globals__, get_p=get_p)
    wrapped = FunctionType(factory.__code__, namespace, factory.__name__,
                           factory.__defaults__, factory.__closure__)
    wrapped.__kwdefaults__ = factory.__kwdefaults__
    return update_wrapper(wrapped, factory)
