"""Bind a qualified XC100 archive to shared GradSCF grids, custom_xc and iKS."""
from dataclasses import dataclass
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
from .pbe import energy_density


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SphericalGrid:
    cartesian: object
    transform: object

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

    Candidate formulas define a CORRECTION to PBE. For this inexpensive pilot,
    parameters must enter affinely; AO directions are constructed once with AD.
    The custom_xc energy input remains an external model definition. No SCF,
    re-inversion, grid reduction or Cartesian orbital-space enlargement occurs.
    """
    fields = frozenset(('n', 'sigma', 'grad', 'lap', 's', 'q', 'u', 'z'))

    def __init__(self, root, system_id='010', *, floor=1e-12, block_size=256):
        started = time.perf_counter()
        self.root, self.system_id = Path(root), str(system_id).zfill(3)
        self.floor, self.block_size = floor, block_size
        folder = self.root/self.system_id
        self.metadata = json.loads((folder/'metadata.json').read_text())
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
        self.target = iks.Target(a['density'], a['overlap'], self.metadata['nelec'])
        self.timings = {'load_seconds': time.perf_counter()-started}

    def inputs(self, physical, fields):
        n = jnp.sqrt(physical.rho**2+self.floor**2)
        result = {'n': n, 'weights': physical.weights}
        if fields & {'sigma', 'grad', 's', 'u'}:
            sigma = jnp.sum(physical.grad_rho**2, axis=-1)
            result.update(sigma=sigma, grad=jnp.sqrt(sigma))
            s2 = sigma/(4*(3*jnp.pi**2)**(2/3)*n**(8/3))
            result.update(s=jnp.sqrt(s2), u=s2/(1+s2))
        if fields & {'lap', 'q', 'z'}:
            lap = physical.laplacian_rho
            q = lap/(4*(3*jnp.pi**2)**(2/3)*n**(5/3))
            result.update(lap=lap, q=q, z=q/jnp.sqrt(1+q*q))
        return result

    def _energy(self, functional, params, density):
        cartesian = self.transform@density@self.transform.T
        return functional.energy_for_density(params, self.context, cartesian)

    def prepare(self):
        started = time.perf_counter()
        def pbe_inputs(physical):
            return {'n': physical.rho, 'sigma': jnp.sum(physical.grad_rho**2, axis=-1),
                    'weights': physical.weights}
        self.baseline_functional = dft.custom_xc(input_fn=pbe_inputs,
            energy_fn=lambda p, x: jnp.sum(x['weights']*energy_density(x['n'], x['sigma'], floor=self.floor)),
            name='external_pbe_reference')
        evaluate = jax.jit(jax.value_and_grad(lambda d: self._energy(self.baseline_functional, None, d)))
        energy, self.vxc = evaluate(self.target.density)
        self.vxc.block_until_ready()
        self.baseline_energy = float(energy)
        if not np.isfinite(self.vxc).all() or not np.isfinite(self.baseline_energy):
            raise ValueError('PBE baseline is nonfinite')
        self.timings['baseline_ad_seconds'] = time.perf_counter()-started
        a = self.arrays
        base = a['kinetic']+a['nuclear']+a['hartree']
        self.problem = iks.WuYang(self.target, a['kinetic'], base, baseline=self.vxc)
        self.reporter = iks.WuYang(self.target, a['kinetic'], base, baseline=self.vxc,
            grid=SphericalGrid(self.grid, self.transform))
        return self

    def bind(self, candidate):
        if (not candidate.affine or len(candidate.parameters) > 2 or
                not candidate.features <= self.fields or set(candidate.initial) & self.fields):
            raise ValueError('This pilot requires an affine correction with <=2 coefficients and declared density features')
        started = time.perf_counter()
        names = [p.name for p in candidate.parameters]
        zeros = {name: jnp.asarray(0.) for name in names}
        if candidate.kind == 'energy':
            functional = candidate.functional(lambda p: self.inputs(p, candidate.features))
            def matrix(params):
                return jax.grad(lambda d: self._energy(functional, params, d))(self.target.density)
        else:
            density = self.transform@self.target.density@self.transform.T
            from gradscf.tools.features import _spin_density_and_gradient, _spin_laplacian
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
        offset = jax.jit(matrix)(zeros)
        directions = jax.jit(jax.jacfwd(matrix))(zeros) if names else {}
        offset.block_until_ready()
        if not all(np.isfinite(x).all() for x in (offset, *directions.values())):
            raise ValueError('Candidate AO correction is nonfinite')
        self.timings['last_candidate_assembly_seconds'] = time.perf_counter()-started
        return {self.system_id: BoundCandidate(self, candidate, offset, directions)}

    @property
    def fingerprint(self):
        files = [Path(__file__), Path(__file__).with_name('pbe.py')]
        record = {'archive': self.archive_hash, 'basis': self.metadata['canonical_basis_sha256'],
            'floor': self.floor, 'block_size': self.block_size, 'baseline': 'PBE+PW92-external',
            'helpers': {p.name: file_hash(p) for p in files}}
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
