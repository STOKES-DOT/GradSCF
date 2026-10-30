#!/usr/bin/env python3
"""Remote one-reference integration check; this is not a designed candidate."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import platform

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import dft
from gradscf.dft import iks
from gradscf_tools.pbe_loop import (AffineTerm, load_teacher_reference,
    build_single_pbe_input, build_molecular_affine_cache, make_pbe_energy_function,
    save_affine_cache, load_affine_cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-json', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-memory-mb', type=float, default=6000.)
    args = parser.parse_args()
    started = perf_counter()
    teacher = load_teacher_reference(args.teacher_json)
    state, enuc = build_single_pbe_input(teacher, max_memory_mb=args.max_memory_mb)
    pbe = make_pbe_energy_function(state, nuclear_repulsion=enuc)
    # A manufactured test with unit prefactor Eh*bohr^3 checks an external
    # current-density callback and its mixed AD. It is not an XC proposal.
    term = dft.custom_xc(input_fn=lambda fields: (fields.rho, fields.weights),
        energy_fn=lambda _, fields: jnp.sum(fields[1]*fields[0]**2),
        name='quadratic_density_interface_probe')
    cache = build_molecular_affine_cache(teacher, state,
        (AffineTerm('quadratic_density_interface_probe', term),), nuclear_repulsion=enuc)
    correction = iks.energy_function(state, term, layout='rks', component='xc')
    energy = lambda params, density: pbe(None, density)+params[0]*correction(None, density)
    weights = {'energy': {'mse':1., 'mae':1.}, 'stationarity': {'norm2':.01}}
    direct = jax.jit(jax.value_and_grad(lambda params: iks.loss(
        teacher.reference, energy, params, weights=weights)[0]))
    cached = jax.jit(jax.value_and_grad(cache.loss))
    theta = jnp.array([1e-4])
    direct_value, direct_gradient = direct(theta)
    cache_value, cache_gradient = cached(theta)
    step = 1e-6
    finite_difference = (direct(theta+step)[0]-direct(theta-step)[0])/(2*step)
    baseline = cache.evaluate(jnp.zeros(1))
    cache_path = Path(args.output).with_suffix('.cache.npz')
    saved = save_affine_cache(cache, cache_path, teacher_json=args.teacher_json,
                              metadata={'purpose':'infra probe only'})
    restored = load_affine_cache(cache_path, expected_sha256=saved['sha256'])
    local_only, _ = build_single_pbe_input(teacher, max_memory_mb=args.max_memory_mb,
                                          with_coulomb=False)
    reused = build_molecular_affine_cache(teacher, local_only,
        (AffineTerm('quadratic_density_interface_probe', term),), nuclear_repulsion=enuc,
        baseline_cache=restored)
    reuse_error = float(jnp.abs(reused.loss(theta)-cache_value))
    valid = bool(baseline.valid and np.allclose(direct_value, cache_value, atol=1e-8, rtol=1e-9)
        and np.allclose(direct_gradient, cache_gradient, atol=1e-7, rtol=1e-8)
        and np.allclose(direct_gradient[0], finite_difference, atol=1e-6, rtol=1e-5)
        and reuse_error < 1e-9 and local_only.rep_tensor is None)
    result = dict(status='passed' if valid else 'failed', candidate_score=False,
        purpose='infra integration only; manufactured test term, no coefficient fit or design round',
        molecule_id=teacher.provenance['molecule_id'], sample_id=teacher.provenance['sample_id'],
        target_kind='CCSD total energy including nuclear repulsion',
        target_total_energy_hartree=float(teacher.target.energy),
        fixed_reference_pbe_energy_hartree=float(baseline.total_energy),
        pbe_energy_error_hartree=float(baseline.energy_error),
        pbe_rotation_norm2=float(baseline.rotation_norm2),
        direct_loss=float(direct_value), cached_loss=float(cache_value),
        direct_parameter_gradient=np.asarray(direct_gradient).tolist(),
        cached_parameter_gradient=np.asarray(cache_gradient).tolist(),
        finite_difference=float(finite_difference), cache_sha256=saved['sha256'],
        baseline_reuse_error=reuse_error, repeated_scf_cycles=0,
        correction_input_has_coulomb_integrals=local_only.rep_tensor is not None,
        preparation_and_probe_seconds=perf_counter()-started,
        node=platform.node(), backend=jax.default_backend(), jax=jax.__version__, dtype='float64')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if not valid:
        raise RuntimeError('Physical PBE/custom callback or mixed-AD integration check failed')


if __name__ == '__main__':
    main()
