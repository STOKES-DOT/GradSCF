"""Four-center molecular CD products, without auxiliary or AO-pair factors.

Cache only the OV Coulomb matrix and selected transition vertices. Their
O((ov)^2 + ov*targets*G) residency must fit the caller's budget. Native
AO2MO supplies these blocks without a complete AO/MO ERI. This is not an
out-of-core or a cache-free integral solver. Large transition equations use
shared GMRES; neither Coulomb rank nor orbital spaces are truncated.
"""
import jax
import jax.numpy as jnp

from .._memory import array_storage_bytes, array_device, workspace_bytes
from ..ao2mo import transform_eri_blocks
from ..solvers import LinearOperator, LinearSolverConfig, solve_complex
from ..solvers.diagnostics import require_converged_derivative
from ..solvers.memory import plan_linear, _largest_fitting
from ._memory import GWExecutionPlan


def _coulomb(source, density, max_memory, reserved_bytes=0):
    if source.get("direct_basis") is not None:
        from ..integrals.backends.native.jk import native_jk_workspace_bytes
        basis = source["direct_basis"]
        primitive, staging = basis, 0
        while hasattr(primitive, "primitive_basis"):
            staging += 8*(6*primitive.nao**2+4*primitive.transform.size)
            primitive = primitive.primitive_basis
        scratch = native_jk_workspace_bytes(primitive, 1, 0.)
        if scratch is None:
            raise TypeError("Conventional direct J requires a complete native topology")
        available = workspace_bytes(max_memory, resident_bytes=reserved_bytes+
            array_storage_bytes(density, *jax.tree.leaves(basis)), device=array_device(density))
        if scratch+staging+48*primitive.nao**2 > available:
            raise ValueError("Conventional native J workspace exceeds the memory budget")
        return basis.get_jk(density, with_k=False, screening_threshold=0.)[0]
    if source.get("eri_pair_matrix") is not None:
        from ..integrals.molecular.jk import build_jk_from_packed
        return build_jk_from_packed(source["eri_pair_matrix"], density,
                                    with_k=False, max_memory=max_memory)[0]
    return jnp.einsum("pqrs,rs->pq", source["eri"], density, precision="highest")


def prepare(coefficients, energy, *, source, nocc, orbs, gi, oi, va, nw,
            fock, hcore, density, max_memory, reserved_bytes, config, v_mf):
    """One fixed orbital frame; energy-dependent dielectric weights stay live."""
    c = jnp.asarray(coefficients)
    if sum(x is not None for x in source.values()) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, direct_basis")
    if jnp.iscomplexobj(c) or any(jnp.iscomplexobj(x) for x in jax.tree.leaves(source)):
        raise NotImplementedError("Conventional molecular GW requires real orbitals and Coulomb data")
    t, pairs = len(oi)*len(va), len(orbs)*len(gi)
    resident = reserved_bytes+array_storage_bytes(*jax.tree.leaves((source, c, energy, fock, hcore, density, v_mf)))
    # J precedes AO2MO: charge a whole array promotion before either runs.
    for value in (source.get("eri"), source.get("eri_pair_matrix")):
        if value is not None:
            promoted = jnp.result_type(c.dtype, density.dtype, value.dtype)
            if value.dtype != promoted:
                resident += value.size*promoted.itemsize
    # Matrix, three ordered vertices, projected W, and loop/update copies.
    resident += 8*(2*t*t+6*t*pairs+2*(nw+1)*pairs+3*c.shape[0]**2+4*energy.size)
    device = array_device(c)
    workspace_bytes(max_memory, resident_bytes=resident, device=device)
    if config is not None and not isinstance(config, LinearSolverConfig):
        raise TypeError("screening_config must be a shared LinearSolverConfig")
    config = config or LinearSolverConfig(method="gmres" if t > 256 else "direct",
                                           rtol=1e-11, atol=1e-13, restart=40)
    working, rhs_batch = 0, 1
    if t and pairs:
        # Direct realification applies the complex operator to 2*t columns;
        # GMRES needs only vector scratch. Shared planning pays for the actual
        # block RHS, residual checks and optional Krylov restart in both cases.
        def estimate(width):
            scratch = (64*t*t if config.method == "direct" else 0)+64*t*width
            return plan_linear(2*t, c.dtype, config, max_memory,
                resident_bytes=resident, operator_min_bytes=scratch,
                nrhs=width, device=device)
        estimate(1)  # Raise invalid config/minimum before integral evaluation.
        def fits(width):
            try:
                estimate(width)
                return True
            except ValueError:
                return False
        rhs_batch = _largest_fitting(1, pairs, fits)
        config, working = estimate(rhs_batch)
    plan = GWExecutionPlan(float(max_memory), resident, working, 1, rhs_batch, True, config)
    potential = (jnp.asarray(fock)-jnp.asarray(hcore)-_coulomb(source, density, max_memory, reserved_bytes=resident-
                     array_storage_bytes(density, *jax.tree.leaves(source)))
                 if v_mf is None else jnp.asarray(v_mf))
    spaces = tuple(c[:, jnp.asarray(ids, dtype=jnp.int32)] for ids in (oi, va, gi, orbs))
    owners = array_storage_bytes(c, energy, fock, hcore, density, v_mf, potential)
    matrix, right, left_i, left_r = transform_eri_blocks(spaces,
        ((0, 1, 0, 1), (0, 1, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1)),
        **source, max_memory=max_memory, reserved_bytes=reserved_bytes+owners)
    right = right.reshape(t, len(gi), len(orbs)).transpose(2, 0, 1)
    left_i = left_i.reshape(len(gi), len(orbs), t).transpose(1, 2, 0)
    left_r = left_r.reshape(len(orbs), len(gi), t).transpose(0, 2, 1)
    cp, co = c[:, jnp.asarray(orbs)], c[:, :nocc]
    exchange, = transform_eri_blocks((cp, co), ((0, 1, 1, 0),), **source,
        max_memory=max_memory, reserved_bytes=reserved_bytes+owners+
        array_storage_bytes(*spaces, matrix, right, left_i, left_r))
    delta = -jnp.einsum("piip->p", exchange)-jnp.einsum("mp,mn,np->p", cp, potential, cp)
    return dict(matrix=matrix.reshape(t, t), right=right, left_i=left_i, left_r=left_r,
        plan=plan, delta_v=jnp.zeros_like(energy).at[jnp.asarray(orbs)].set(delta))


def product(matrix, weights, left, right, config):
    """C_L^T (I-DV)^-1 D C_R; ordered raw inputs are not symmetrized."""
    if not weights.size or not right.size:
        return jnp.zeros(right.shape[1:], dtype=jnp.result_type(left, right, weights))
    def apply(x):
        weighted = weights if x.ndim == 1 else weights[:, None]
        return x-weighted*((matrix @ x.real)+1j*(matrix @ x.imag))
    op = LinearOperator(matrix.shape, jnp.result_type(matrix, weights), apply, matmat=apply)
    solved = solve_complex(op, weights[:, None]*right, config=config)
    value = jnp.sum(left*solved.solution, axis=0)
    return require_converged_derivative(value, solved.converged)


def screened(preparation, gap, frequencies):
    """Each RHS tile is reduced immediately, retaining only projected W."""
    matrix, plan = preparation["matrix"], preparation["plan"]
    left, right = preparation["left_i"], preparation["right"]
    p, t, g = right.shape
    if not t or not p*g:
        return jnp.zeros((len(frequencies), p, g), dtype=right.dtype)
    left, right = left.transpose(1, 0, 2).reshape(t, -1), right.transpose(1, 0, 2).reshape(t, -1)
    block = plan.rhs_batch
    def frequency(w):
        weights = -4*gap/(w*w+gap*gap)
        def body(index, output):
            start = jnp.minimum(index*block, p*g-block)
            lhs = jax.lax.dynamic_slice_in_dim(left, start, block, axis=1)
            rhs = jax.lax.dynamic_slice_in_dim(right, start, block, axis=1)
            value = product(matrix, weights, lhs, rhs, plan.screening_config).real
            return jax.lax.dynamic_update_slice_in_dim(output, value, start, axis=0)
        return jax.lax.fori_loop(0, (p*g+block-1)//block, jax.checkpoint(body),
                                jnp.zeros(p*g, dtype=right.dtype)).reshape(p, g)
    return jax.lax.map(jax.checkpoint(frequency), frequencies)


def residue(omega, ctx):
    from .self_energy import _static_residue_weights
    data = ctx["transition"]
    weights = _static_residue_weights(omega, ctx["mo_energy"], ctx["ef"], ctx["occupation_sign"])
    left, right = ctx["c_left"], ctx["c_right"]
    gap, eta = data["gap"], ctx["eta"]
    if not gap.size or not right.shape[1]:
        return jnp.asarray(0j, dtype=jnp.complex128)
    def pole(index):
        def active(_):
            frequency = jnp.abs(ctx["mo_energy"][index]-omega)
            d = 2*(1/(frequency-gap+2j*eta)-1/(frequency+gap))
            return product(data["matrix"], d, left[:, index, None],
                           right[:, index, None], ctx["execution_plan"].screening_config)[0]
        # Dynamic indices need take rather than Python slices under tracing.
        return weights[index]*jax.lax.cond(weights[index] != 0, active,
                                          lambda _: jnp.asarray(0j, dtype=jnp.complex128), None)
    # Keep inactive bare poles away from the linear solver, including in AD.
    def body(index, value):
        return value+jax.checkpoint(pole)(index)
    value = jax.lax.fori_loop(0, right.shape[1], body, jnp.asarray(0j, dtype=jnp.complex128))
    return value-jnp.sum(weights*ctx["wmn_static"])


def g0w0_conventional(*, mo_energy, mo_coeff, nocc, eri, eri_pair_matrix, direct_basis,
                      fock_matrix, hcore_matrix, density_matrix, nw, eta, orbs,
                      diff_mode, qp_solver, mo_energy_poles, screening_energy,
                      g_orbitals, screening_occupied, screening_virtual, linearized,
                      evaluate_only, qp_initial, qp_tol, max_memory, reserved_bytes,
                      screening_config, v_mf_matrix, _preparation=None):
    from .g0w0 import _requested_orbitals, _window_indices, _qp_loop
    from .outer_response import require_occupation_gap
    from .freq import scaled_legendre_grid
    from .types import GWResult
    if jnp.iscomplexobj(mo_energy) or jnp.iscomplexobj(mo_coeff):
        raise NotImplementedError("Conventional molecular GW requires real inputs")
    e, c = jnp.asarray(mo_energy, dtype=jnp.float64), jnp.asarray(mo_coeff, dtype=jnp.float64)
    if not 0 < nocc < e.size:
        raise ValueError("Restricted GW requires occupied and virtual orbitals")
    orbs = _requested_orbitals(orbs, e.size)
    gi = _window_indices(g_orbitals, range(e.size), "g_orbitals")
    oi = _window_indices(screening_occupied, range(nocc), "screening_occupied")
    va = _window_indices(screening_virtual, range(nocc, e.size), "screening_virtual")
    poles = e if mo_energy_poles is None else jnp.asarray(mo_energy_poles)
    screen = poles if screening_energy is None else jnp.asarray(screening_energy)
    if poles.shape != e.shape or screen.shape != e.shape:
        raise ValueError("G/W spectra must match the full MO spectrum")
    require_occupation_gap(poles, nocc, "Conventional GW G")
    require_occupation_gap(screen, nocc, "Conventional GW W")
    data = _preparation if _preparation is not None else prepare(c, e,
        source=dict(eri=eri, eri_pair_matrix=eri_pair_matrix, direct_basis=direct_basis),
        nocc=nocc, orbs=orbs, gi=gi, oi=oi, va=va, nw=nw, fock=fock_matrix,
        hcore=hcore_matrix, density=density_matrix, max_memory=max_memory,
        reserved_bytes=reserved_bytes, config=screening_config, v_mf=v_mf_matrix)
    gap = (screen[jnp.asarray(va, dtype=jnp.int32)][None, :]-screen[jnp.asarray(oi, dtype=jnp.int32)][:, None]).reshape(-1)
    frequencies, wts = scaled_legendre_grid(nw)
    w = data.get("projected")
    if w is None:
        w = screened(data, gap, jnp.concatenate((jnp.zeros(1), frequencies)))
    prepared = dict(wmn_p=w[1:].transpose(1, 0, 2), wmn_static=w[0],
                    c_left=data["left_r"], c_right=data["right"], p_index=jnp.asarray(orbs))
    qp, sigma, mask, done, residual, weight, diagnostics = _qp_loop(
        mo_energy=poles, b_mn=None, channels=(), wmn=None, freqs=frequencies,
        wts=wts, ef=.5*(jnp.max(poles[:nocc])+jnp.min(poles[nocc:])), eta=eta,
        delta_v=data["delta_v"], nocc=nocc, orbs=orbs, diff_mode=diff_mode,
        qp_solver=qp_solver, e_mf=e, linearized=linearized, evaluate_only=evaluate_only,
        g_orbitals=gi, qp_initial=qp_initial, qp_tol=qp_tol, prepared=prepared,
        execution_plan=data["plan"], return_weight=True, return_diagnostics=True,
        transition=dict(matrix=data["matrix"], gap=gap))
    return GWResult(mo_energy=qp, mo_coeff=c, converged=done, sigma_qp=sigma,
        converged_mask=mask, nw=nw, qp_residual=residual, qp_weight=weight,
        qp_computed_mask=jnp.zeros_like(e, dtype=bool).at[jnp.asarray(orbs)].set(not evaluate_only),
        screening_energy=screen, g_orbitals=gi, screening_occupied=oi,
        screening_virtual=va, method="g0w0", **diagnostics)
