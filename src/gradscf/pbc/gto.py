"""Three-dimensional neutral periodic Gaussian cells, independent of PySCF."""
from dataclasses import dataclass
from functools import lru_cache
import json
from importlib.resources import files
import numpy as np
import jax.numpy as jnp
from ..data.molecule import parse_molecule_spec, ANGSTROM_TO_BOHR
from .kpoints import reciprocal_vectors, uniform_kpoints


@lru_cache(maxsize=1)
def _gth_data():
    return json.loads(files('gradscf.integrals.periodic').joinpath('gth_data.json').read_text())


@dataclass
class Cell:
    atom: object
    a: object
    basis: object = 'gth-szv'
    pseudo: object = 'gth-pade'
    unit: str = 'Angstrom'
    mesh: tuple = (31,31,31)
    charge: int = 0
    spin: int = 0
    precision: float = 1e-9
    dimension: int = 3
    cart: bool = True

    def build(self):
        if self.dimension!=3 or self.charge!=0 or not self.cart:
            raise NotImplementedError('Initial PBC implementation requires neutral 3D Cartesian cells.')
        if not 0<self.precision<1:
            raise ValueError('precision must be between zero and one.')
        raw=np.asarray(self.a,dtype=float)
        if raw.shape!=(3,3) or not np.isfinite(raw).all() or np.linalg.det(raw)<=0:
            raise ValueError('a must be a finite right-handed nonsingular 3x3 lattice.')
        mesh=np.asarray(self.mesh)
        if mesh.shape!=(3,) or np.any(mesh<3) or np.any(mesh!=mesh.astype(int)) or np.any(mesh%2==0):
            raise ValueError('Use three odd FFT mesh sizes >=3 to retain +/-G symmetry.')
        self.mesh=tuple(map(int,mesh))
        self.spec=parse_molecule_spec(self.atom,unit=self.unit,charge=self.charge,spin=self.spin)
        scale=ANGSTROM_TO_BOHR if self.unit.lower().startswith(('ang','a')) else 1.
        self.lattice=jnp.asarray(raw*scale)
        self.coords=jnp.asarray(self.spec.coords_bohr)
        fractional=np.asarray(self.coords)@np.linalg.inv(np.asarray(self.lattice))
        delta=fractional[:,None]-fractional[None,:]
        delta=delta-np.rint(delta)
        distances=np.linalg.norm(delta@np.asarray(self.lattice),axis=-1)
        if np.any(distances[np.triu_indices(len(fractional),1)]<1e-8):
            raise ValueError('Atoms coincide modulo a lattice translation.')
        data=_gth_data()
        self.basis_spec={}
        self.pseudopotentials=[]
        for symbol in self.spec.symbols:
            value=self.basis.get(symbol) if isinstance(self.basis,dict) else self.basis
            if isinstance(value,str) and value.lower() in data['basis']:
                value=data['basis'][value.lower()].get(symbol)
                if not value: raise ValueError(f'No bundled {self.basis} basis for {symbol}.')
            self.basis_spec[symbol]=value
            pp=self.pseudo.get(symbol) if isinstance(self.pseudo,dict) else self.pseudo
            if isinstance(pp,str):
                pp=data['pseudo'].get(pp.lower(),{}).get(symbol)
            if not pp: raise ValueError(f'Provide supported GTH parameters for {symbol}.')
            if pp[1]<=0 or len(pp[3])>4 or len(pp[5:])!=pp[4]:
                raise ValueError('Invalid GTH local potential or projector channels.')
            for radius,count,coupling in pp[5:]:
                if radius<=0 or (count and np.asarray(coupling).shape!=(count,count)):
                    raise ValueError('Invalid GTH projector radius or coupling matrix.')
            self.pseudopotentials.append(pp)
        self.charges=jnp.asarray([sum(pp[0]) for pp in self.pseudopotentials])
        self.nelectron=int(sum(sum(pp[0]) for pp in self.pseudopotentials))
        if abs(self.spin)>self.nelectron or (self.nelectron+self.spin)%2:
            raise ValueError('spin must be compatible with the valence electron count.')
        self.nelec=((self.nelectron+self.spin)//2,(self.nelectron-self.spin)//2)
        from ..integrals.basis import prepare_basis
        self.topology,self.parameters=prepare_basis(self.spec,self.basis_spec)
        self._version=getattr(self,'_version',0)+1
        return self

    @property
    def volume(self):
        return jnp.linalg.det(self.lattice)

    def reciprocal_vectors(self):
        return reciprocal_vectors(self.lattice)

    def make_kpts(self,mesh):
        return uniform_kpoints(self.lattice,mesh)


def M(**kwargs):
    return Cell(**kwargs).build()
