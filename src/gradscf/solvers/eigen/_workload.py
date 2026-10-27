"""Opt-in, scalar-only runtime accounting for Davidson operator actions.

The ordered host callback intervals include diagnostic synchronization and
callback overhead. They are not clean kernel timings. No callbacks, norms, or
host clocks are added when a solver's ``workload_callback`` is ``None``.

``columns`` is the submitted static width; ``effective_columns`` counts the
packed, nonzero columns (including columns later discarded at an RPA subspace
capacity boundary). Initialization uses iteration zero; Ritz iterations start
at one. ``locked_roots`` is zero because neither solver hard-locks roots.
Converged-root counts refer to the residual tolerance; independent structure
and derivative guards remain outside this workload accounting.
Record volume is bounded by the configured iteration limit; physical vectors
are never copied to the host.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import isfinite
from time import perf_counter

import jax
import jax.numpy as jnp


def _tree_norm(values):
    return jnp.sqrt(sum(jnp.sum(jnp.abs(value) ** 2)
                        for value in jax.tree.leaves(values)))


class _DavidsonWorkload:
    def __init__(self, callback: Callable[[dict], None], *, solver: str,
                 nroots: int, max_subspace: int, tolerance):
        if not callable(callback):
            raise TypeError("workload_callback must be callable or None")
        self.callback = callback
        self.solver = solver
        self.nroots = nroots
        self.max_subspace = max_subspace
        self.tolerance = tolerance
        self._started = None
        self._action_starts = {}

    def __call__(self, event):
        self.callback(event)

    def _deliver(self, iteration, effective_columns, subspace_dim,
                 residual_norms, action_norm, tolerance, *, event, purpose, columns):
        now = perf_counter()
        iteration = None if iteration is None else int(iteration)
        if event == "apply_start" and purpose == "initial":
            # A compiled diagnostic solve can be invoked more than once.
            self._started = now
            self._action_starts.clear()
        if self._started is None:
            self._started = now
        key = (iteration, purpose)
        elapsed = 0.0
        if event == "apply_start":
            self._action_starts[key] = now
        elif event == "apply_end":
            elapsed = now - self._action_starts.pop(key, now)
        residuals = [float(value.real) for value in residual_norms.tolist()]
        converged = sum(isfinite(value) and value <= float(tolerance.real)
                        for value in residuals)
        self.callback({
            "solver": self.solver,
            "event": event,
            "purpose": purpose,
            "iteration": iteration,
            "columns": columns,
            "effective_columns": int(effective_columns),
            "subspace_dim": None if subspace_dim is None else int(subspace_dim),
            "subspace_capacity": self.max_subspace,
            "requested_roots": self.nroots,
            "active_roots": self.nroots - converged,
            "converged_roots": converged,
            # These solvers retain converged Ritz vectors but do not hard-lock.
            "locked_roots": 0,
            "residual_norms": residuals,
            "action_norm": float(action_norm),
            "elapsed_seconds": elapsed,
            "total_elapsed_seconds": now - self._started,
            "timing_kind": "ordered_host_callbacks",
        })

    def record(self, event, purpose, *, iteration, columns, effective_columns,
               subspace_dim, residual_norms, action_norm=0.0):
        payload = (iteration, effective_columns, subspace_dim,
                   residual_norms, action_norm, self.tolerance)
        jax.debug.callback(
            partial(self._deliver, event=event, purpose=purpose, columns=columns),
            *jax.tree.map(jax.lax.stop_gradient, payload),
            ordered=True,
        )

    def apply(self, function, *values, iteration, purpose, effective_columns,
              subspace_dim, residual_norms):
        metadata = dict(iteration=iteration, columns=values[0].shape[1],
                        effective_columns=effective_columns,
                        subspace_dim=subspace_dim, residual_norms=residual_norms)
        self.record("apply_start", purpose, action_norm=_tree_norm(values), **metadata)
        result = function(*values)
        self.record("apply_end", purpose, action_norm=_tree_norm(result), **metadata)
        return result


def _workload_reporter(callback, *, solver, nroots, max_subspace, tolerance):
    """Reuse a reporter across a numerical solve and its final verification.

    Pass this result as the numerical solver's ``workload_callback`` and use
    its ``record``/``apply`` methods for the caller's final action. An absent
    callback returns ``None``; unknown iteration/subspace metadata uses ``None``.
    """
    if callback is None or isinstance(callback, _DavidsonWorkload):
        return callback
    return _DavidsonWorkload(
        callback, solver=solver, nroots=nroots, max_subspace=max_subspace,
        tolerance=tolerance,
    )
