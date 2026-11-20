"""Bind a qualified XC100 archive to shared GradSCF grids, custom_xc and iKS."""
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import dft
from gradscf.dft import iks
from gradscf.integrals.basis import basis_from_molecule_spec
from gradscf.integrals.grids.blocks import AOGrid
from gradscf.data.molecule import parse_molecule_spec
from gradscf.scf.molecules import QuadratureGrid
from gradscf.tools.features import _spin_density_and_gradient, _spin_laplacian
from .pbe import energy_density


_Q_NODES, _Q_WEIGHTS = np.polynomial.legendre.leggauss(96)
_q_nodes = (_Q_NODES + 1) / 2
_Q_NODES, _Q_WEIGHTS = _q_nodes**3, 1.5*_q_nodes**2*_Q_WEIGHTS


def _density_fields(rho, gradient, laplacian, fields, floor):
    n = jnp.sqrt(rho**2 + floor**2)
    result = {'n': n}
    if fields & {'sigma', 'grad', 's', 'u'}:
        sigma = jnp.sum(gradient**2, axis=-1)
        s2 = sigma / (4*(3*jnp.pi**2)**(2/3)*n**(8/3))
        result.update(sigma=sigma, grad=jnp.sqrt(sigma), s=jnp.sqrt(s2), u=s2/(1+s2))
    if fields & {'lap', 'q', 'z'}:
        q = laplacian / (4*(3*jnp.pi**2)**(2/3)*n**(5/3))
        result.update(lap=laplacian, q=q, z=q/jnp.sqrt(1+q*q))
    return result


def q_path_energy_density(candidate, params, rho, gradient, laplacian, floor):
    """Integrate rho * v_xc[t*rho] on the straight path from zero density."""
    nodes = jnp.asarray(_Q_NODES, dtype=rho.dtype)
    weights = jnp.asarray(_Q_WEIGHTS, dtype=rho.dtype)

    def add(index, total):
        t = nodes[index]
        fields = _density_fields(t*rho, t*gradient, t*laplacian, candidate.features, floor)
        return total + weights[index]*candidate.evaluate(params, fields)

    return rho*jax.lax.fori_loop(0, len(_Q_NODES), add, jnp.zeros_like(rho))


@partial(jax.jit, static_argnames=('candidate',))
def _q_path_energy(grid, density, candidate, params, floor):
    fields = candidate.features
    needs_laplacian = bool(fields & {'lap', 'q', 'z'})
    needs_gradient = needs_laplacian or bool(fields & {'sigma', 'grad', 's', 'u'})
    deriv = 2 if needs_laplacian else int(needs_gradient)

    def add(total, ao, derivative, ao_laplacian, coords, weights):
        if needs_gradient:
            rho, gradient = _spin_density_and_gradient(ao, derivative, density)
        else:
            rho = jnp.sum(ao*(ao@density.T), axis=1)
            gradient = jnp.zeros((), dtype=rho.dtype)
        laplacian = (_spin_laplacian(ao, derivative, ao_laplacian, density)
                     if needs_laplacian else jnp.zeros((), dtype=rho.dtype))
        energy = q_path_energy_density(candidate, params, rho, gradient, laplacian, floor)
        return total + jnp.dot(weights, energy)

    return grid.reduce(add, jnp.zeros((), dtype=density.dtype), deriv=deriv)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SphericalGrid:
    cartesian: object
    transform: object

    def tree_flatten(self): return (self.cartesian,self.transform),None

    @classmethod
    def tree_unflatten(cls, metadata, children): return cls(*children)

    @property
    def nao(self): return self.transform.shape[1]

    @property
    def weights(self): return self.cartesian.weights

    def reduce(self, function, initial, **kwargs):
        t = self.transform
        def add(total, ao, derivative, laplacian, coordinates, weights):
            return function(total, ao@t, None if derivative is None else derivative@t,
                None if laplacian is None else laplacian@t, coordinates, weights)
        return self.cartesian.reduce(add, initial, **kwargs)


@dataclass(frozen=True)
class GridContext:
    grid_ao: object
    atom_coords: object
    atom_charges: object

    @property
    def ao(self): return jnp.zeros((0, self.grid_ao.nao))

    @property
    def grid(self): return QuadratureGrid(weights=self.grid_ao.weights, coords=self.grid_ao.coords)


@dataclass(frozen=True)
class BoundCandidate:
    case: object
    candidate: object
    offset: object
    directions: object

    def potential(self, params):
        value = self.case.vxc+self.offset
        for name, matrix in self.directions.items(): value = value+params[name]*matrix
        return value

    def __call__(self, params): return self.case.problem.score(self.potential(params))

    def diagnose(self, params): return self.case.reporter.score(self.potential(params))


class XC100Case:
    """Real closed-shell acceptance sample; all archive conventions remain fixed.

    Legacy candidates define affine corrections to PBE. Open bindings also
    accept general local corrections and complete local XC potentials.
    """
    fields = frozenset(('n', 'sigma', 'grad', 'lap', 's', 'q', 'u', 'z'))

    def __init__(self, root, system_id='010', *, floor=1e-12, block_size=256, max_parameters=4,
                 occupation_tolerance=1e-8):
        started = time.perf_counter()
        self.root, self.system_id = Path(root), str(system_id).zfill(3)
        self.floor, self.block_size = floor, block_size
        if max_parameters is not None and (not isinstance(max_parameters,int) or max_parameters<1):
            raise ValueError('max_parameters must be positive or None for an open model')
        self.max_parameters = max_parameters
        self.occupation_tolerance = occupation_tolerance
        folder = self.root/self.system_id
        self.metadata = json.loads((folder/'metadata.json').read_text())
        self.metadata_hash = file_hash(folder/'metadata.json')
        if self.metadata['spin'] != 0 or self.metadata['status'] != 'passed':
            raise ValueError('Require a qualified closed-shell archive')
        self.archive_hash = file_hash(folder/'density.npz')
        if self.archive_hash != self.metadata['artifact_sha256']:
            raise ValueError('Archive checksum differs from the qualified release')
        if file_hash(self.root/'canonical_basis.json') != self.metadata['canonical_basis_sha256']:
            raise ValueError('Explicit basis checksum differs from the qualified release')
        with np.load(folder/'density.npz', allow_pickle=False) as data:
            self.arrays = {key: data[key] for key in data.files}
        a = self.arrays
        self.basis_definitions = json.loads((self.root/'canonical_basis.json').read_text())
        self.symbols = self.metadata['symbols']
        self.atom_string = '; '.join(f'{s} {x:.17g} {y:.17g} {z:.17g}'
            for s, (x, y, z) in zip(self.symbols, a['atom_coords_bohr']))
        spec = parse_molecule_spec(self.atom_string, unit='Bohr',
                                   charge=self.metadata['charge'], spin=0)
        basis = basis_from_molecule_spec(spec, basis=self.basis_definitions)
        self.grid = AOGrid.from_basis(basis, a['grid_coords_bohr'], a['grid_weights'], block_size=block_size)
        self.context = GridContext(self.grid, a['atom_coords_bohr'], a['atom_charges'])
        self.transform = jnp.asarray(a['spherical_to_cartesian'])
        if self.transform.shape != (self.grid.nao, a['density'].shape[0]):
            raise ValueError('Saved angular embedding and reconstructed AO basis differ')
        self.target = iks.Target(a['density'], a['overlap'], self.metadata['nelec'],
                                 occupation_tolerance=occupation_tolerance)
        self.timings = {'load_seconds': time.perf_counter()-started}

    def inputs(self, physical, fields):
        gradient = physical.grad_rho if fields & {'sigma', 'grad', 's', 'u'} else None
        laplacian = physical.laplacian_rho if fields & {'lap', 'q', 'z'} else None
        result = _density_fields(physical.rho, gradient, laplacian, fields, self.floor)
        return dict(result, weights=physical.weights)

    def energy_report(self, candidate, params, density):
        cartesian = self.transform@density@self.transform.T
        energy = float(_q_path_energy(self.grid, cartesian, candidate, params, self.floor))
        if not np.isfinite(energy):
            raise ValueError(f'Nonfinite Q-path XC energy for {self.system_id}')
        return {'xc_energy_hartree': energy}

    def _energy(self, functional, params, density):
        cartesian = self.transform@density@self.transform.T
        return functional.energy_for_density(params, self.context, cartesian)

    def prepare(self, *, baseline=None, include_energy=True):
        started = time.perf_counter()
        def pbe_inputs(physical):
            return {'n': physical.rho, 'sigma': jnp.sum(physical.grad_rho**2, axis=-1),
                    'weights': physical.weights}
        self.baseline_functional = dft.custom_xc(input_fn=pbe_inputs,
            energy_fn=lambda p, x: jnp.sum(x['weights']*energy_density(x['n'], x['sigma'], floor=self.floor)),
            name='external_pbe_reference')
        if baseline is None:
            def evaluate(density):
                return self._energy(self.baseline_functional, None, density)

            if include_energy:
                energy, self.vxc = jax.jit(jax.value_and_grad(evaluate))(self.target.density)
            else:
                energy = None
                self.vxc = jax.jit(jax.grad(evaluate))(self.target.density)
        else:
            energy, potential = baseline
            self.vxc = jnp.asarray(potential)
            if self.vxc.shape != self.target.density.shape:
                raise ValueError('Cached baseline shape differs from the target')
        self.vxc.block_until_ready()
        self.baseline_energy = None if energy is None else float(energy)
        if not np.isfinite(self.vxc).all() or (energy is not None and not np.isfinite(self.baseline_energy)):
            raise ValueError('PBE baseline is nonfinite')
        self.timings['baseline_ad_seconds'] = time.perf_counter()-started
        a = self.arrays
        base = a['kinetic']+a['nuclear']+a['hartree']
        self.problem = iks.WuYang(self.target, a['kinetic'], base, baseline=self.vxc)
        self.reporter = iks.WuYang(self.target, a['kinetic'], base, baseline=self.vxc,
            grid=SphericalGrid(self.grid, self.transform))
        return self

    def candidate_matrix(self, candidate):
        """Return the existing uncompiled AO map for separate compile/run timing."""
        if (not candidate.affine or (self.max_parameters is not None and len(candidate.parameters) > self.max_parameters) or
                not candidate.features <= self.fields or set(candidate.initial) & self.fields):
            raise ValueError(f'This pilot requires an affine correction with <={self.max_parameters} coefficients and declared density features')
        if candidate.kind == 'energy':
            functional = candidate.functional(lambda p: self.inputs(p, candidate.features))
            def matrix(params):
                return jax.grad(lambda d: self._energy(functional, params, d))(self.target.density)
        else:
            density = self.transform@self.target.density@self.transform.T
            def matrix(params):
                def add(total, ao, derivative, laplacian, coords, weights):
                    n, gradient = _spin_density_and_gradient(ao, derivative, density)
                    sigma = jnp.sum(gradient**2, axis=-1)
                    n = jnp.sqrt(n*n+self.floor**2)
                    lap = _spin_laplacian(ao, derivative, laplacian, density)
                    s2 = sigma/(4*(3*jnp.pi**2)**(2/3)*n**(8/3))
                    q = lap/(4*(3*jnp.pi**2)**(2/3)*n**(5/3))
                    values = candidate.evaluate(params, dict(n=n, sigma=sigma, grad=jnp.sqrt(sigma),
                        lap=lap, s=jnp.sqrt(s2), q=q, u=s2/(1+s2), z=q/jnp.sqrt(1+q*q)))
                    return total+jnp.einsum('g,gi,gj->ij', weights*values, ao, ao)
                cartesian = self.grid.reduce(add, jnp.zeros((self.grid.nao,)*2), deriv=2)
                return self.transform.T@cartesian@self.transform
        return matrix

    def bind(self, candidate, *, complete_potential=False):
        if self.max_parameters is None:
            from gradscf.model.autoxc.bindings import bind_local_potential
            from .ingredients import features_for_candidate
            baseline = jnp.zeros_like(self.vxc) if complete_potential else self.vxc
            bound=bind_local_potential(self.problem,self.grid,candidate,
                features_for_candidate(self,candidate.features),baseline=baseline,
                reporter=self.reporter,transform=self.transform,max_memory=8000.)
            self.timings['cached_affine']=bound.cached_affine
            return {self.system_id:bound}
        from .direct import GGA_FIELDS, bind_potential
        if candidate.kind=='potential' and candidate.features<=GGA_FIELDS:
            return bind_potential(self,candidate)
        started = time.perf_counter()
        matrix = self.candidate_matrix(candidate)
        names = [p.name for p in candidate.parameters]
        zeros = {name: jnp.asarray(0.) for name in names}
        offset = jax.jit(matrix)(zeros)
        directions = jax.jit(jax.jacfwd(matrix))(zeros) if names else {}
        offset.block_until_ready()
        if not all(np.isfinite(x).all() for x in (offset, *directions.values())):
            raise ValueError('Candidate AO correction is nonfinite')
        self.timings['last_candidate_assembly_seconds'] = time.perf_counter()-started
        return {self.system_id: BoundCandidate(self, candidate, offset, directions)}

    @property
    def fingerprint(self):
        files = [Path(__file__).with_name(name) for name in
                 ('molecule.py','pbe.py','checks.py','direct.py','reference.py')]
        if self.max_parameters is None:files.append(Path(__file__).with_name('ingredients.py'))
        record = {'archive': self.archive_hash, 'metadata': self.metadata_hash,
            'basis': self.metadata['canonical_basis_sha256'], 'max_parameters': self.max_parameters,
            'occupation_tolerance': self.occupation_tolerance,
            'floor': self.floor, 'block_size': self.block_size, 'baseline': 'PBE+PW92-external',
            'helpers': {p.name: file_hash(p) for p in files}}
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
