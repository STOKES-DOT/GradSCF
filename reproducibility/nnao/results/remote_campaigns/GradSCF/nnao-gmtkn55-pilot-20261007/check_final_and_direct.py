"""Supplement the pilot with final-point AD and exact shell-direct energies."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from validate_nnao_pilot import (atomic_json, directional_check, file_hash,
    numerical_runtime, record_directory, verify_source_imports)


def main():
    run = Path(sys.argv[1])
    data = Path(sys.argv[2])
    jax, experiment_class = numerical_runtime('cpu')
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from jax.flatten_util import ravel_pytree
    from gradscf.data.molecule import atomic_number
    from gradscf.model.nnao import MACEBasisModel, build_graph

    structures = {r['structure_id']: r for r in
                  map(json.loads, data.read_text().splitlines())}
    report = dict(backend=jax.default_backend(), dtype='float64',
        imported_sources=verify_source_imports(ROOT), results=[])
    destination = run / 'final_and_direct.json'
    for identifier in ('W4-11/h2', 'W4-11/ch4', 'W4-11/hf'):
        start = time.perf_counter()
        directory = record_directory(run / 'validation', identifier)
        previous = json.loads((directory / 'result.json').read_text())
        if previous['status'] != 'passed':
            raise RuntimeError(f'{identifier}: main pilot did not pass')
        geometry = structures[identifier]; geometry['name'] = identifier
        cache = Path(previous['cache_path'])
        if file_hash(cache) != previous['cache_sha256']:
            raise RuntimeError('Cache changed since main pilot')
        df = experiment_class(geometry=geometry, basis_family='szp663_direct',
            core_primitives=6, jk_backend='df', integral_cache=cache)
        with np.load(directory / 'checkpoint.npz', allow_pickle=False) as saved:
            vector = saved['parameters']; final_outputs = saved['outputs']

        model = MACEBasisModel(**previous['model_config'], rngs=nnx.Rngs(previous['seed']))
        graph = build_graph([atomic_number(s) for s in df.symbols], df.coords,
                            element_order=model.elements)
        definition, parameters, other = nnx.split(model, nnx.Param, ...)
        _, unravel = ravel_pytree(parameters)

        @jax.jit
        def predict(x):
            return nnx.merge(definition, unravel(x), other)(graph)

        @jax.jit
        def backward(x, cotangent):
            return jax.vjp(predict, x)[1](cotangent)[0]

        def evaluate(x):
            energy, gradient, info = df.evaluate(predict(jnp.asarray(x)))
            return energy, np.asarray(backward(jnp.asarray(x), gradient)), info

        np.testing.assert_allclose(predict(jnp.asarray(vector)), final_outputs,
                                   atol=1e-12, rtol=1e-12)
        final_energy, gradient, info = evaluate(vector)
        np.testing.assert_allclose(final_energy, previous['final_energy_hartree'],
                                   atol=1e-9, rtol=0)
        direction = np.random.default_rng(31).normal(size=vector.shape)
        direction /= np.linalg.norm(direction)
        fd = directional_check(evaluate, vector, gradient, direction,
                               step=1e-4, atol=2e-6, rtol=2e-5)
        if not fd['passed'] or not np.isfinite(gradient).all():
            raise RuntimeError(f'{identifier}: final parameter FD failed')

        direct = experiment_class(geometry=geometry, basis_family='szp663_direct',
                                 core_primitives=6, jk_backend='direct')
        forward = jax.jit(direct._implicit_value)
        comparisons = []
        for label, outputs, df_energy in (
            ('initial', df.layout.reference_outputs(), previous['initial_energy_hartree']),
            ('after_5_adam_steps', final_outputs, final_energy)):
            energy, diagnostics = forward(jnp.asarray(outputs), direct.ps, direct.ph, direct.rep)
            diagnostics = {key: bool(value) if key == 'converged' else
                           int(value) if key == 'scf_cycles' else float(value)
                           for key, value in diagnostics.items()}
            if not diagnostics['converged'] or diagnostics['orbital_residual'] > 1e-7:
                raise RuntimeError(f'{identifier}: direct SCF is not stationary')
            if not np.isfinite(energy) or diagnostics['reconstruction_error'] > 1e-9:
                raise RuntimeError(f'{identifier}: invalid direct energy')
            comparisons.append(dict(point=label, df_energy_hartree=df_energy,
                direct_energy_hartree=float(energy),
                df_minus_direct_hartree=df_energy - float(energy), diagnostics=diagnostics))
        result = dict(structure_id=identifier, final_parameter_directional_check=fd,
            final_scf=info, exact_direct_comparisons=comparisons,
            elapsed_seconds=time.perf_counter() - start)
        report['results'].append(result)
        atomic_json(destination, report)
        print(f'{identifier}: final FD and direct comparison complete', flush=True)
        jax.clear_caches()
    report['passed'] = len(report['results'])
    atomic_json(destination, report)


if __name__ == '__main__':
    main()
