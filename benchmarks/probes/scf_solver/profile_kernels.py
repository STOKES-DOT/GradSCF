"""Fixed-input GKS kernel diagnostics; called after the real SCF, never a loop.

Arrays stay dynamic JIT arguments. Timings at different compilation boundaries
cannot be summed to partition the production SCF. No ERI is copied or saved.
"""
from __future__ import annotations

import json
from pathlib import Path
import time


def run_kernels(inputs, result, cfg, output: Path, budget_seconds):
    """Append JSONL records and return them; the outer process owns the hard limit.

    This budget is checked between operations and cannot preempt a compilation
    or a running kernel. Fewer than three warm samples means partial_budget.
    Compiler memory_analysis is buffer accounting, never measured process RSS.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gradscf.integrals.molecular.jk import build_jk_from_packed, _packed_jk_workspace
    from gradscf.scf.gks import (GKSConfig, _generalized_xc, _hermitian,
                                 _spin_diagonal, _validate_mode,
                                 generalized_energy_and_fock, generalized_jk)
    from gradscf.solvers.scf import orbital_gradient_norm, orbital_orthogonalizer

    started = time.perf_counter()
    deadline = started + max(0., float(budget_seconds))
    rows = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("a", buffering=1)

    def emit(row):
        row.update(elapsed_seconds=time.perf_counter()-started,
                   remaining_seconds=max(0., deadline-time.perf_counter()))
        stream.write(json.dumps(row, allow_nan=False)+"\n")
        stream.flush()
        rows.append(row)

    def metadata(value):
        return [dict(shape=list(a.shape), dtype=str(a.dtype), nbytes=int(a.nbytes))
                for a in jax.tree_util.tree_leaves(value) if hasattr(a, "shape")]

    def snapshot(value):
        # Only kernel outputs (scalars or <= spinor matrices), never ERI/AOs.
        snapshots = []
        for a in jax.tree_util.tree_leaves(value):
            host = np.asarray(a)
            entry = dict(shape=list(host.shape), dtype=str(host.dtype),
                         finite=bool(np.all(np.isfinite(host))))
            if entry["finite"] and host.size:
                entry.update(norm=float(np.linalg.norm(host.reshape(-1))),
                             max_abs=float(np.max(np.abs(host))))
                if host.size == 1:
                    scalar = host.reshape(-1)[0]
                    entry["value"] = ({"real": float(scalar.real), "imag": float(scalar.imag)}
                                      if np.iscomplexobj(host) else float(scalar))
            snapshots.append(entry)
        return snapshots

    def difference(got, reference, tolerance=1e-8):
        delta = np.asarray(got)-np.asarray(reference)
        finite = bool(np.all(np.isfinite(delta)))
        maximum = float(np.max(np.abs(delta))) if finite else None
        return dict(max_abs=maximum, rms=float(np.sqrt(np.mean(np.abs(delta)**2)))
                    if finite else None, atol=tolerance, rtol=0.,
                    passed=bool(finite and maximum <= tolerance))

    def probe(name, function, args, *, reserve=15., warm=10, cost=None, check=None):
        if deadline-time.perf_counter() < reserve:
            emit(dict(event="kernel", name=name, status="skipped_budget", reserve_seconds=reserve))
            return None
        row = dict(event="kernel", name=name, status="started", inputs=metadata(args),
                   warm_target=warm, cost_estimate=cost, all_array_arguments_dynamic=True)
        emit(dict(row))
        phase = "lower"
        try:
            jax.block_until_ready(args)
            tick = time.perf_counter()
            lowered = jax.jit(function).lower(*args)
            row["lower_seconds"] = time.perf_counter()-tick
            emit(dict(event="kernel_phase", name=name, phase=phase, seconds=row["lower_seconds"]))
            phase = "compile"
            tick = time.perf_counter()
            executable = lowered.compile()
            row["compile_seconds"] = time.perf_counter()-tick
            emit(dict(event="kernel_phase", name=name, phase=phase, seconds=row["compile_seconds"]))
            try:
                analysis = executable.memory_analysis()
                fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                          "temp_size_in_bytes", "generated_code_size_in_bytes",
                          "host_argument_size_in_bytes", "host_output_size_in_bytes",
                          "host_temp_size_in_bytes")
                row["compiler_memory_not_rss"] = {key: int(getattr(analysis, key))
                    for key in fields if getattr(analysis, key, None) is not None}
            except Exception as exc:
                row["compiler_memory_error"] = repr(exc)
            phase = "first_execute"
            tick = time.perf_counter()
            value = jax.block_until_ready(executable(*args))
            row["first_execute_seconds"] = time.perf_counter()-tick
            emit(dict(event="kernel_phase", name=name, phase=phase, seconds=row["first_execute_seconds"]))
            phase = "warm_execute"
            samples = []
            for _ in range(warm):
                if deadline-time.perf_counter() < max(1., row["first_execute_seconds"]*1.1):
                    break
                tick = time.perf_counter()
                value = jax.block_until_ready(executable(*args))
                samples.append(time.perf_counter()-tick)
                emit(dict(event="kernel_phase", name=name, phase=phase, sample=len(samples), seconds=samples[-1]))
            row.update(warm_seconds=samples, warm_samples=len(samples),
                       warm_median_seconds=float(np.median(samples)) if samples else None,
                       status="ok" if len(samples) >= 3 else "partial_budget")
            phase = "snapshot_and_consistency"
            row["snapshot"] = snapshot(value)
            if check is not None:
                row["consistency"] = check(value)
                if any(not item["passed"] for item in row["consistency"].values()):
                    row["status"] = "consistency_failed"
            emit(row)
            return value
        except Exception as exc:
            row.update(status="error", phase=phase, error=repr(exc))
            emit(row)
            return None

    try:
        if cfg.get("method", "GKS") != "GKS":
            emit(dict(event="kernels", status="not_applicable", reason="GKS boundaries only"))
            return rows
        config = GKSConfig(xc_spec=cfg["xc"], max_cycle=cfg.get("max_cycle", 80),
            conv_tol=cfg.get("conv_tol", 1e-10),
            conv_tol_density=cfg.get("density_tol", 1e-8),
            conv_tol_grad=cfg.get("gradient_tol", 1e-7),
            collinear=cfg.get("collinear", "col"), max_memory=cfg.get("max_memory", 4000.),
            grid_block_size=-1)
        kind = _validate_mode(config)
        density, fock, coeff, occ = (result.density_matrix, result.fock_matrix,
                                    result.mo_coeff, result.mo_occ)
        n, m, grid = inputs.overlap.shape[0], density.shape[0], inputs.ao.shape[0]
        eri = inputs.eri
        common = dict(nao=n, spinor_dimension=m, ngrid=grid,
                      eri_stored_bytes=int(eri.nbytes), spinor_matrix_bytes=int(density.nbytes),
                      note="Logical sizes/scaling estimates, not measured traffic or hardware FLOPs")
        emit(dict(event="kernel_scope", status="ok", budget_seconds=float(budget_seconds),
            config=dict(xc=config.xc_spec, collinear=config.collinear), dimensions=common,
            snapshot_converged=bool(result.converged), snapshot_cycles=int(result.cycles),
            note="Returned snapshot reused without changing its density; independent JIT boundaries; "
                 "first execution is after explicit compile; no production-loop additive attribution"))
        emit(dict(event="kernel", name="diis_extrapolate", status="not_profiled_missing_history",
            reason="GKSResult retains no DIIS ring history; no representative history synthesized"))
        emit(dict(event="kernel", name="ao2mo", status="not_applicable",
            reason="Generalized SCF has no four-index AO2MO boundary"))
        if eri.ndim not in (1, 2):
            emit(dict(event="kernel", name="packed_jk", status="not_applicable", reason="ERI is not packed"))
            return rows
        spatial = (density[:n, :n]+density[n:, n:]).T
        blocks = density.reshape(2, n, 2, n).transpose(0, 2, 1, 3)
        plans = {}
        for label, d, with_j, with_k in (("j", spatial, True, False), ("k", blocks, False, True)):
            widths = _packed_jk_workspace(eri, d, with_j, with_k, None, 1)
            plans[label] = dict(portable_widths_j_p_q=list(widths),
                contraction_count=((n*(n+1)//2+widths[0]-1)//widths[0] if with_j
                                   else ((n+widths[1]-1)//widths[1])*((n+widths[2]-1)//widths[2])))
        emit(dict(event="packed_jk_plan", status="ok", eri_layout="s4" if eri.ndim == 2 else "s8",
            eri_elements=int(eri.size), backend=jax.default_backend(), default_workspace_mb=64.,
            j_gather_required=bool(eri.ndim == 1 or eri.size >= 2**31), plans=plans,
            note="No max_memory/block_size override; s4 CPU uses portable(1); widths describe portable path. "
                 "Only s8 float64 CPU may dispatch native; facade max_memory does not override this call"))
        j = probe("packed_j_only", lambda e, d: build_jk_from_packed(e, d, with_k=False),
            (eri, spatial), reserve=60., warm=3,
            cost=dict(common, packed_pair_flops_estimate=4*(n*(n+1)//2)**2))
        k = probe("packed_k_only_four_spin_blocks", lambda e, d:
            build_jk_from_packed(e, d, with_j=False), (eri, blocks), reserve=120., warm=3,
            cost=dict(common, dense_equivalent_flops_estimate=16*n**4))
        tuned_widths = _packed_jk_workspace(eri, blocks, False, True, 2048., 16)
        probe("packed_k_only_workspace_2048mb_block16", lambda e, d:
            build_jk_from_packed(e, d, with_j=False, max_memory=2048., block_size=16),
            (eri, blocks), reserve=120., warm=3,
            cost=dict(common, max_memory_mb=2048., block_size_ceiling=16,
                      portable_widths_j_p_q=list(tuned_widths),
                      note="Diagnostic scheduling/workspace contrast; identical ERI and density; no E2E inference"),
            check=(lambda v: {"K_vs_default": difference(v[1], k[1])}) if k is not None else None)
        xc = probe("generalized_xc", lambda d, ao, deriv, w:
            _generalized_xc(d, ao, deriv, w, config, kind),
            (density, inputs.ao, inputs.ao_deriv1, inputs.grid_weights), reserve=60., warm=3,
            cost=dict(common, scaling="O(ngrid*nao^2), density/gradient + XC + Vxc integration"),
            check=lambda v: {"xc_energy_Ha": difference(v[0], result.xc_energy)})
        x = probe("spinor_orthogonalization", lambda s: _spin_diagonal(
            orbital_orthogonalizer(s, config.orthogonalization_eps)), (inputs.overlap,),
            cost=dict(common, scaling="O(nao^3), spatial Cholesky/fallback eigen + spin duplication"),
            check=lambda x: {"whitening": difference(x.conj().T @ result.overlap_matrix @ x,
                                                     jnp.eye(m, dtype=x.dtype))})
        if x is not None:
            def diagonalize(f, x):
                e, c = jnp.linalg.eigh(_hermitian(x.conj().T @ f @ x))
                return e, x @ c
            probe("spinor_diagonalization", diagonalize, (fock, x), reserve=30.,
                cost=dict(common, scaling="O(spinor_dimension^3), transform + Hermitian eigh + backtransform"),
                check=lambda ec: {"generalized_eigen_residual": difference(
                    fock @ ec[1], (result.overlap_matrix @ ec[1])*ec[0], tolerance=1e-7)})
            probe("ao_diis_commutator", lambda f, d, s, x:
                x.conj().T @ (f @ d @ s-s @ d @ f) @ x,
                (fock, density, result.overlap_matrix, x),
                cost=dict(common, dense_complex_flops_estimate=48*m**3))
        probe("density_projection", lambda c, o: _hermitian((c*o) @ c.conj().T), (coeff, occ),
            cost=dict(common, dense_complex_flops_estimate=8*m**3),
            check=lambda d: {"density_vs_result": difference(d, density)})
        probe("mo_orbital_residual", orbital_gradient_norm, (fock, coeff, occ),
            cost=dict(common, dense_complex_flops_estimate=16*m**3),
            check=lambda g: {"gradient_vs_result": difference(g, result.orbital_gradient_norm)})
        jk = probe("generalized_jk", generalized_jk, (eri, density), reserve=120., warm=3,
                   cost=dict(common, dense_equivalent_flops_estimate=20*n**4))
        def energy_fock(d, h, e, enuc, ao, deriv, w):
            return generalized_energy_and_fock(density=d, hcore=h, eri=e,
                nuclear_repulsion=enuc, ao=ao, ao_deriv1=deriv, grid_weights=w, config=config)
        total = probe("generalized_energy_and_fock", energy_fock,
            (density, result.hcore_matrix, eri, jnp.asarray(inputs.nuclear_repulsion),
             inputs.ao, inputs.ao_deriv1, inputs.grid_weights), reserve=180., warm=3,
            cost=dict(common, scaling="packed J + four spin K + O(ngrid*nao^2) XC"),
            check=lambda e: {"energy_Ha": difference(e[0], result.total_energy),
                             "xc_energy_Ha": difference(e[1], result.xc_energy),
                             "fock_Ha": difference(e[2], fock)})
        consistency = {}
        if jk is not None and j is not None:
            consistency["J_public_vs_generalized"] = difference(_hermitian(_spin_diagonal(j[0])), jk[0])
        if jk is not None and k is not None:
            exchange = k[1].transpose(0, 2, 1, 3).reshape(m, m)
            consistency["K_public_vs_generalized"] = difference(_hermitian(exchange), jk[1])
        if jk is not None and xc is not None and total is not None:
            reconstructed = _hermitian(result.hcore_matrix+jk[0]-result.exact_exchange_fraction*jk[1]+xc[1])
            consistency["Fock_components_vs_total"] = difference(reconstructed, total[2])
        emit(dict(event="kernel_cross_consistency", checks=consistency,
            status="consistency_failed" if any(not v["passed"] for v in consistency.values()) else "ok"))
    except Exception as exc:
        emit(dict(event="kernels", status="error", error=repr(exc)))
    finally:
        stream.close()
    return rows
