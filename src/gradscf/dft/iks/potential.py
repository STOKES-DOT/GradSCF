"""Local multiplicative potential maps over the existing checkpointed AO grid."""
from dataclasses import dataclass, replace
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._memory import DEFAULT_MAX_MEMORY_MB, array_device, tree_storage_bytes, validate_memory_mb, workspace_bytes
from gradscf.integrals.grids.blocks import ao_point_bytes
from .types import CHANNELS, _check, _components, _layout


def _assemble(panels, layout):
    if layout == 'rks': return panels[0]
    if layout == 'uks': return panels
    scalar, x, y, z = panels
    return jnp.block([[scalar+z, x-1j*y], [x+1j*y, scalar-z]])


def constant_potential(overlap, value, *, layout='rks'):
    """Analytic c*S gauge: two constants for fixed-population UKS, one otherwise.

    A uniform magnetic field is NOT discarded as a spinor gauge. The spatial
    AO overlap is supplied independently of quadrature-overlap error.
    """
    _layout(layout)
    s, value = jnp.asarray(overlap), jnp.asarray(value)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError('overlap must be a spatial square matrix')
    if jnp.iscomplexobj(value) or value.shape != ((2,) if layout == 'uks' else ()):
        raise ValueError('A constant is real and scalar, or two separate UKS constants')
    _check(jnp.all(jnp.isfinite(value)), 'A constant potential must be finite')
    if layout == 'uks': return value[:, None, None]*s
    return value*(jnp.kron(jnp.eye(2, dtype=s.dtype), s) if layout == 'gks' else s)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Potential:
    """basis(params, coordinates) -> (block_points,nparam) real local functions.

    Coefficients have shape (nparam,), (2,nparam) or (4,nparam) for RKS, UKS,
    or scalar/x/y/z GKS fields. apply and its physical transpose share the
    same grid/basis; no npotential*nao*nao tensor is stored. The evaluator must
    be block-local. Arbitrary evaluator internals are not an RSS guarantee.
    """

    grid: object
    overlap: object
    basis: object
    nparam: int
    params: object = None
    layout: str = 'rks'
    max_memory: float = DEFAULT_MAX_MEMORY_MB

    def __post_init__(self):
        _layout(self.layout)
        if not callable(self.basis) or isinstance(self.nparam, bool) or not isinstance(self.nparam, Integral) or self.nparam <= 0:
            raise ValueError('A potential needs a callable basis and positive nparam')
        s = jnp.asarray(self.overlap)
        if s.shape != (self.grid.nao, self.grid.nao) or jnp.iscomplexobj(s):
            raise ValueError('Local potential AO grid requires a real spatial overlap with matching shape')
        object.__setattr__(self, 'overlap', s)
        object.__setattr__(self, 'max_memory', validate_memory_mb(self.max_memory))

    def tree_flatten(self):
        return (self.grid, self.overlap, self.params), (self.basis, self.nparam, self.layout, self.max_memory)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        obj = object.__new__(cls)
        for name, value in zip(('grid', 'overlap', 'params', 'basis', 'nparam', 'layout', 'max_memory'),
                               (*children, *metadata)):
            object.__setattr__(obj, name, value)
        return obj

    def _plan(self, operand):
        channels, n = CHANNELS[self.layout], self.grid.nao
        dtype = np.empty((), dtype=jnp.result_type(
            operand, self.overlap, *jax.tree.leaves((self.grid, self.params)))).real.dtype
        # Retained provider/parameters/operand plus output, accumulators and
        # physical spatial/spinor assembly. Evaluator activations stay external.
        resident = tree_storage_bytes(self.grid, self.overlap, self.params, operand)
        device = array_device(operand, self.overlap)
        fixed_elements = 8*channels*n*n+2*channels*self.nparam
        available = workspace_bytes(self.max_memory,
            resident_bytes=resident+fixed_elements*dtype.itemsize, device=device)
        point_elements = self.nparam+channels+2*n+4
        if available < ao_point_bytes(self.grid, deriv=0)+dtype.itemsize*point_elements:
            raise ValueError('Potential memory budget cannot hold one quadrature point')
        features = jax.eval_shape(self.basis, self.params,
                                 jax.ShapeDtypeStruct((1, 3), self.grid.coords.dtype))
        if getattr(features, 'shape', None) != (1, self.nparam) or np.dtype(features.dtype).kind == 'c':
            raise ValueError('Potential basis must return real (block_points,nparam) values')
        dtype = np.dtype(jnp.result_type(dtype, features.dtype))
        size = dtype.itemsize
        available = workspace_bytes(self.max_memory, resident_bytes=resident+fixed_elements*size,
                                    device=device)
        per_point = ao_point_bytes(self.grid, deriv=0)+size*point_elements
        width = min(self.grid.block_size, available//per_point)
        if width < 1:
            raise ValueError('Potential memory budget cannot hold one quadrature point')
        return replace(self.grid, block_size=int(width)), dtype

    def _features(self, coords, weights):
        value = jnp.asarray(self.basis(self.params, coords))
        if value.shape != (coords.shape[0], self.nparam) or jnp.iscomplexobj(value):
            raise ValueError('Potential basis must return real (block_points,nparam) values')
        value = jnp.where(weights[:, None] != 0, value, 0.)
        valid = jnp.all(jnp.isfinite(value))
        _check(valid, 'Potential basis must be finite on active grid points')
        return value*jnp.where(valid, 1., jnp.nan)

    def matrix(self, coefficients, *, constant=None):
        coefficients = jnp.asarray(coefficients)
        channels = CHANNELS[self.layout]
        shape = (self.nparam,) if self.layout == 'rks' else (channels, self.nparam)
        if coefficients.shape != shape or jnp.iscomplexobj(coefficients):
            raise ValueError(f'{self.layout} real coefficients require shape {shape}')
        grid, dtype = self._plan(coefficients)
        c = coefficients.reshape(channels, self.nparam)
        initial = jnp.zeros((channels, grid.nao, grid.nao), dtype=dtype)
        def add(total, ao, deriv, lap, coords, weights):
            values = self._features(coords, weights) @ c.T
            return total+jnp.einsum('g,gi,gj,gc->cij', weights, ao, ao, values, precision='highest')
        result = _assemble(grid.reduce(add, initial), self.layout)
        if constant is not None:
            result = result+constant_potential(self.overlap, constant, layout=self.layout)
        return result

    def moments(self, density):
        """Physical dual: sum(b*moments(D)) = Re Tr(U(b) D)."""
        density = jnp.asarray(density)
        blocks = _components(density, self.grid.nao, self.layout)
        grid, dtype = self._plan(density)
        initial = jnp.zeros((CHANNELS[self.layout], self.nparam), dtype=dtype)
        def add(total, ao, deriv, lap, coords, weights):
            rho = jnp.einsum('gi,cij,gj->gc', ao, blocks, ao, precision='highest')
            return total+jnp.einsum('g,gc,gp->cp', weights, rho, self._features(coords, weights), precision='highest')
        value = grid.reduce(add, initial)
        return value[0] if self.layout == 'rks' else value
