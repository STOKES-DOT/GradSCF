"""Reproduce and correct the frozen 2.75-Angstrom branch with native GradSCF."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['GRADSCF_NATIVE_THREADS'] = '4'
from pathlib import Path
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.initial_guess import atomic_guess_data, atomic_density_guess

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'nnao-ethane-dissociation-20261019'


def prepare(index):
    row = json.loads((SOURCE / 'structures.json').read_text())[index]
    definition = json.loads((OUT / 'pool-definition.json').read_text())
    layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
        basis_family=definition['basis_family'], core_primitives=definition['core_primitives'], cart=False)
    layout = bind_pool_candidate(layout, definition['candidate'])
    saved = np.load(SOURCE / f'distance-{index:03d}/best-coefficients.npz')
    coefficients = np.array(saved['mace_coefficients'])
    frozen = next(json.loads(s) for s in (SOURCE / f'distance-{index:03d}/frozen-bases.jsonl').read_text().splitlines()
                  if json.loads(s)['basis'] == 'MACE-update3351')
    shells = layout.atom_shells(layout.bind(jnp.asarray(coefficients)))
    for actual, expected in zip(shells, frozen['atom_shells']):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert a[0] == b[0]
            np.testing.assert_allclose(a[1:], b[1:], atol=1e-12, rtol=1e-12)
    experiment, manifest = prepare_fixed_pool(row, layout, OUT / f'cache-{index:03d}/pool.npz',
        storage='device', auxbasis='def2-universal-jkfit', implicit_tolerance=1e-8)
    return row, layout, saved, experiment, manifest


def atomic_seed(experiment, layout, coefficients):
    overlap = experiment.project(jnp.asarray(coefficients), *experiment.raw)[0]
    return atomic_density_guess(overlap, *atomic_guess_data(layout))


if __name__ == '__main__':
    started = time.perf_counter()
    row, layout, saved, experiment, manifest = prepare(25)
    c = jnp.asarray(saved['mace_coefficients'])
    print('PREPARED', row['distance_angstrom'], layout.topology.nao, manifest['pool_nao'], flush=True)
    high, _, high_info = experiment.evaluate_device(c, gradient=False)
    print('HCORE', high, high_info, flush=True)
    seed = atomic_seed(experiment, layout, c)
    low, gradient, low_info = experiment.evaluate_device(c, init_density=seed)
    report = dict(distance_angstrom=row['distance_angstrom'], hcore_energy_hartree=high,
        atomic_energy_hartree=low, energy_lowering_hartree=high-low,
        gradient_norm=float(np.linalg.norm(gradient)), gradient_finite=bool(np.isfinite(gradient).all()),
        hcore_scf=high_info, atomic_scf=low_info, seconds=time.perf_counter()-started,
        backend=jax.default_backend(), jax=jax.__version__, finite_differences=False)
    (OUT / 'native-branch-probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print('RESULT', json.dumps(report), flush=True)
