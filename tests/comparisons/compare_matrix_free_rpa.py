"""Opt-in 300-dimensional matrix-free real RPA response and finite-difference check.

Run with PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python
tests/comparisons/compare_matrix_free_rpa.py. No molecular/GPU scaling claim.
"""

import json, time, platform, resource, sys, shlex
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as j
import numpy as np
from gradscf.solvers import LinearOperator, EigenSolverConfig, solve_rpa

n = 300
d = j.linspace(1.0, 4.0, n)
u = j.asarray(np.random.default_rng(41).normal(size=(n, 2)) * 0.002)
cfg = EigenSolverConfig(
    nroots=2, max_subspace=8, atol=1e-9, gradient_mode="implicit_eigenvector"
)


def calc(t):
    factor = u * t

    def apply(x):
        return (d * x if x.ndim == 1 else d[:, None] * x) + factor @ (factor.T @ x)

    a = LinearOperator(
        (n, n), j.float64, apply, diagonal=d + j.sum(factor**2, axis=1), matmat=apply
    )
    b = LinearOperator(
        (n, n),
        j.float64,
        lambda x: 0.05 * x,
        diagonal=j.full((n,), 0.05),
        matmat=lambda x: 0.05 * x,
    )
    out = solve_rpa(a, b, config=cfg)
    return out.values.sum() + j.sum(out.x[:3] ** 2), out


start = time.perf_counter()
(value, result), grad = jax.jit(jax.value_and_grad(calc, has_aux=True))(1.0)
jax.block_until_ready(grad)
elapsed = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_bytes = peak if sys.platform == "darwin" else peak * 1024
fd = (calc(1.0001)[0] - calc(0.9999)[0]) / 0.0002
assert np.all(np.asarray(result.converged & result.response_valid))
assert np.isfinite(float(grad))
np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=2e-5)
print(
    json.dumps(
        dict(
            command="PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 "
            + shlex.join([sys.executable, *sys.argv]),
            platform=platform.platform(),
            process_peak_rss_bytes=peak_bytes,
            jax=jax.__version__,
            backend=jax.default_backend(),
            dtype="float64",
            n=n,
            seed=41,
            rank=2,
            max_subspace=8,
            elapsed_compile_and_first_grad_seconds=elapsed,
            value=float(value),
            gradient=float(grad),
            finite_difference=float(fd),
            gradient_error=float(abs(grad - fd)),
            energies=np.asarray(result.values).tolist(),
            residuals=np.asarray(result.residual_norms).tolist(),
            converged=np.asarray(result.converged).tolist(),
            response_valid=np.asarray(result.response_valid).tolist(),
            stability_certified=bool(result.stability_certified),
        ),
        indent=2,
    )
)
