"""RHF reference problems for active-pool calibration and free contraction.

All electrons participate in SCF. Only the target valence/polarization pool is
uncontracted; fixed core basis functions are retained. No MACE is required.
"""
from dataclasses import replace
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals, scf
from gradscf.integrals.molecular.density_fitting import project_factors, unpack_factors
from ..basis import prepare_direct_basis
from ..initialization import with_pool_parameters
from ..integrals.pool import active_pool_basis, pool_projection
from .objective import BasisRHF, _df_rhf_solver
from .trainer import Adam


def overlap_diagnostics(overlap, *, max_condition=1e10):
    eigenvalues = np.linalg.eigvalsh(np.asarray(overlap))
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    condition = largest/smallest if smallest > 0 else float('inf')
    if not np.isfinite(condition) or smallest <= 1e-10 or condition > max_condition:
        raise ValueError(f'Ill-conditioned active pool overlap: min={smallest}, condition={condition}.')
    return dict(pool_min_overlap_eigenvalue=smallest, pool_condition_number=condition)


class PoolRHF:
    """CPU native integrals with explicit CPU/GPU implicit RHF response."""
    def __init__(self, row, keys, *, basis_family='szp663_direct', core_primitives=6,
                 auxbasis='def2-universal-jkfit', implicit_tolerance=1e-8,
                 max_condition=1e10, scf_device=None):
        if row.get('charge') != 0 or row.get('spin') != 0:
            raise NotImplementedError('Pool pilot supports neutral spin=0 RHF references only.')
        self.row = row
        self.keys = tuple(keys)
        self.max_condition = max_condition
        self.cpu_device = jax.devices('cpu')[0]
        self.scf_device = self.cpu_device if scf_device is None else scf_device
        atom = list(zip(row['symbols'], row['coords_angstrom']))
        with jax.default_device(self.cpu_device):
            self.layout = prepare_direct_basis(atom, cart=False, basis_family=basis_family,
                                           core_primitives=core_primitives)
            if self.layout.nelectron % 2:
                raise ValueError('Closed-shell RHF requires an even electron count.')
            top, p = active_pool_basis(self.layout)
            self.pool_nao = top.nao
            self.plan = integrals.make_plan(top, backend='native')
            at, ap = integrals.prepare_basis(atom, auxbasis, cart=False)
            self.aux = integrals.make_auxiliary_plan(top, at)
            self.metric = self.aux.metric_factor(p, ap)
            self.enuc = scf.nuclear_repulsion_energy(p.nuclear_coords, jnp.asarray(top.nuclear_charges))
        self.value, self.gradient = _df_rhf_solver(self.layout.nelectron, implicit_tolerance, .2)

        def raw(parameters):
            moved = with_pool_parameters(self.layout, parameters, self.keys)
            _, bound = active_pool_basis(moved)
            # Geometry/auxiliary basis are fixed for this reference problem.
            bound = replace(bound, centers=p.centers, nuclear_coords=p.nuclear_coords)
            s = self.plan.evaluate('overlap', bound)
            h = self.plan.evaluate('kinetic', bound)+self.plan.evaluate('nuclear', bound)
            b = self.aux.factors(bound, ap, metric_factor=self.metric)
            return s, h, b

        def physical(parameters):
            s, h, b = raw(parameters)
            return s, h, unpack_factors(b, top.nao)

        self.raw = jax.jit(raw)
        self.physical = jax.jit(physical)
        self.pullback = jax.jit(lambda parameters, cotangent:
            jax.vjp(physical, parameters)[1](cotangent)[0])

    def evaluate(self, parameters, *, gradient=True):
        start = time.perf_counter()
        parameters = jax.device_put(parameters, self.cpu_device)
        physical = jax.block_until_ready(self.physical(parameters))
        condition = overlap_diagnostics(physical[0], max_condition=self.max_condition)
        physical = jax.device_put(physical, self.scf_device)
        enuc = jax.device_put(self.enuc, self.scf_device)
        prepared = time.perf_counter()
        if gradient:
            (value, info), cotangent = self.gradient(physical, enuc)
        else:
            value, info = self.value(physical, enuc)
        details = BasisRHF._checked_info(value, info)
        solved = time.perf_counter()
        derivative = np.asarray(self.pullback(parameters,
            jax.device_put(cotangent, self.cpu_device))) if gradient else None
        if derivative is not None and not np.isfinite(derivative).all():
            raise RuntimeError('Nonfinite pool exponent gradient.')
        details.update(condition, structure_id=self.row['structure_id'],
            energy_hartree=float(value), pool_nao=self.pool_nao,
            contracted_nao=self.layout.topology.nao,
            scf_backend=self.scf_device.platform,
            integral_seconds=prepared-start, scf_seconds=solved-prepared,
            backward_seconds=time.perf_counter()-solved)
        return float(value), derivative, details

    def prepare_contraction(self, parameters):
        with jax.default_device(self.cpu_device):
            parameters = jax.device_put(parameters, self.cpu_device)
            layout = with_pool_parameters(self.layout, parameters, self.keys)
            raw = jax.block_until_ready(self.raw(parameters))
        diagnostics = overlap_diagnostics(raw[0], max_condition=self.max_condition)
        raw, enuc = jax.device_put((raw, self.enuc), self.scf_device)
        return _DeviceFreeContractionRHF(layout, raw, enuc, self.value, self.gradient,
                                       self.row['structure_id'], diagnostics)


class FreeContractionRHF:
    """Independent radial coefficients at fixed exponents and target AO count."""
    def __init__(self, layout, raw, enuc, value, gradient, identifier, diagnostics, *, atomic_guess=None):
        self.layout, self.raw, self.enuc = layout, raw, enuc
        self.value, self.gradient = value, gradient
        self.identifier, self.diagnostics = identifier, diagnostics
        def physical(c, s, h, b):
            t = pool_projection(layout, c)
            return t.T@s@t, t.T@h@t, project_factors(b, t)
        self.project = jax.jit(physical)

        def energy(c, s, h, b, init_density=None):
            inputs=physical(c,s,h,b)
            if init_density is not None:
                return value(inputs,enuc,atomic_guess,init_density=init_density)
            return value(inputs,enuc) if atomic_guess is None else value(inputs,enuc,atomic_guess)
        self._value = jax.jit(energy)
        self._value_grad = jax.jit(jax.value_and_grad(energy, argnums=0, has_aux=True))

    def evaluate_device(self, coefficients, *, gradient=True, init_density=None):
        """Keep the coefficient gradient on device for the MACE pullback.

        A single executable consumes the fixed cache once for projection, the
        existing stable implicit SCF, and contraction AD. Exponents are closed
        over by the layout; the native dynamic-exponent path remains separate.
        """
        start = time.perf_counter()
        args=(coefficients,*self.raw)
        if init_density is not None:
            args=args+(jax.lax.stop_gradient(jnp.asarray(init_density)),)
        if gradient:
            (energy, info), derivative = self._value_grad(*args)
        else:
            energy, info = self._value(*args)
            derivative = None
        details = BasisRHF._checked_info(energy, info)
        if derivative is not None and not bool(jnp.all(jnp.isfinite(derivative))):
            raise RuntimeError('Nonfinite free-contraction gradient.')
        details.update(self.diagnostics, structure_id=self.identifier,
            energy_hartree=float(energy), contracted_nao=self.layout.topology.nao,
            evaluation_seconds=time.perf_counter()-start)
        return float(energy), derivative, details

    def evaluate(self, coefficients, *, gradient=True, init_density=None):
        energy, derivative, details = self.evaluate_device(coefficients, gradient=gradient,init_density=init_density)
        return energy, np.asarray(derivative) if derivative is not None else None, details


class _DeviceFreeContractionRHF(FreeContractionRHF):
    """Stage-B coefficients arrive from the host optimizer or CPU layout."""
    def evaluate_device(self, coefficients, *, gradient=True, init_density=None):
        if init_density is not None:
            init_density=jax.device_put(init_density,self.enuc.device)
        return super().evaluate_device(jax.device_put(coefficients, self.enuc.device),
                                        gradient=gradient,init_density=init_density)


def minimize_reference(initial, evaluate, *, max_steps=20, learning_rate=.01,
                       gradient_tolerance=1e-5, max_backtracks=6, bounds=None, resume=None):
    """Small reference optimization with checked, energy-decreasing Adam steps.

    A shortened trial uses the same proposed Adam moments, accepted only after
    its own converged energy and finite gradient have been evaluated. Failure
    leaves the previously accepted parameters and moments intact.
    """
    if isinstance(max_steps, bool) or max_steps < 0 or max_backtracks < 0:
        raise ValueError('Optimization step/backtrack counts must be nonnegative.')
    if not np.isfinite([learning_rate, gradient_tolerance]).all() or min(learning_rate, gradient_tolerance) <= 0:
        raise ValueError('Learning rate and gradient tolerance must be finite and positive.')
    x = np.array(initial, dtype=np.float64, copy=True)
    optimizer = Adam(learning_rate)
    state = optimizer.init(x)
    energy, gradient, info = evaluate(x)
    if not np.isfinite(energy) or not np.isfinite(gradient).all():
        raise RuntimeError('Invalid initial reference energy/gradient.')
    initial_energy = energy
    history = [dict(step=0, accepted=True, energy_hartree=energy,
                    gradient_norm=float(np.linalg.norm(gradient)))]
    status, accepted, first_step = 'max_steps', 0, 1
    if resume is not None:
        if not np.array_equal(x, resume['parameters']):
            raise ValueError('Resume parameters differ from the supplied accepted checkpoint.')
        if (not np.isclose(energy, resume['energy_hartree'], atol=1e-9, rtol=0)
                or not np.allclose(gradient, resume['gradient'], atol=1e-7, rtol=1e-5)):
            raise ValueError('Resume reference energy/gradient changed; check sources and SCF branch.')
        state = optimizer.decode(resume['optimizer'], optimizer.init(x))
        energy, gradient = resume['energy_hartree'], np.array(resume['gradient'], copy=True)
        initial_energy = resume['initial_energy_hartree']
        history = list(resume['history'])
        accepted = resume['accepted_steps']
        first_step = history[-1]['step']+1
        if first_step > max_steps+1:
            raise ValueError('Resume budget precedes the saved optimization step.')
    for step in range(first_step, max_steps+1):
        if np.linalg.norm(gradient) <= gradient_tolerance:
            status = 'converged'
            break
        proposed, proposed_state = optimizer.update(x, gradient, state)
        proposed = np.asarray(proposed)
        if bounds is not None:
            proposed = np.clip(proposed, *bounds)
        committed = False
        failures = []
        for backtrack in range(max_backtracks+1):
            trial = x+(proposed-x)*(.5**backtrack)
            try:
                value, derivative, details = evaluate(trial)
                if (not np.isfinite(value) or not np.isfinite(derivative).all()
                        or value > energy+1e-12):
                    raise RuntimeError('Nonfinite or energy-increasing reference candidate.')
            except Exception as error:
                failures.append(str(error))
                continue
            x, energy, gradient, info, state = trial, value, derivative, details, proposed_state
            accepted += 1
            committed = True
            history.append(dict(step=step, accepted=True, energy_hartree=energy,
                gradient_norm=float(np.linalg.norm(gradient)), backtracks=backtrack))
            break
        if not committed:
            status = 'rejected'
            history.append(dict(step=step, accepted=False, errors=failures))
            break
    if np.linalg.norm(gradient) <= gradient_tolerance:
        status = 'converged'
    return dict(parameters=x, gradient=np.asarray(gradient), info=info, history=history,
                optimizer=optimizer.encode(state), status=status, accepted_steps=accepted,
                initial_energy_hartree=initial_energy, energy_hartree=float(energy),
                gradient_norm=float(np.linalg.norm(gradient)),
                optimizer_converged=status == 'converged')
