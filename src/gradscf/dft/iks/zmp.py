"""Finite-penalty Zhao--Morrison--Parr inversion using shared GradSCF solvers.

The Coulomb callback maps ONE spatial AO density to its Coulomb matrix. It
may use native direct J/K, packed ERI or DF; no integral representation is
materialized here. RKS uses lambda*J[dD]; UKS uses 2*lambda*J[dD_sigma].
See Zhao et al., Phys. Rev. A50,2138 (1994), and KSPies zmp.py (Apache2.0).
"""
from dataclasses import dataclass
from math import isfinite

import jax
import jax.numpy as jnp

from gradscf._memory import DEFAULT_MAX_MEMORY_MB, tree_storage_bytes, workspace_bytes
from gradscf.solvers import EigenSolverConfig, solve_hermitian
from gradscf.solvers.eigen.spectral import inverse_sqrt, occupied_projector
from gradscf.solvers.nonlinear import NonlinearConfig, iterate_eager
from .orbital import OrbitalReference
from .response import density_norm2
from .types import _check


@dataclass(frozen=True)
class ZMPStage:
    penalty: float
    iterations: int
    scf_residual: float
    coulomb_energy: float
    converged: bool


@dataclass(frozen=True)
class ZMPResult:
    target: object
    reference: object
    density: object
    fock: object
    correction_potential: object
    stages: tuple
    coulomb_energy: float
    scf_residual: float
    gap: float
    converged: bool
    valid: bool
    density_error: object = None
    relative_density_error: object = None
    xc_potential: object = None
    method: str = 'zmp'


def invert(target, base_fock, *, coulomb, lambdas=(1.,5.,10.,20.,50.,100.),
           grid=None, hcore=None, initial=None, coulomb_tolerance=1e-6,
           density_tolerance=1e-5, scf_tolerance=1e-9, gap_tolerance=1e-7,
           maxiter=400, diis_space=12, damping=.2, max_memory=DEFAULT_MAX_MEMORY_MB):
    """Offline real RKS/UKS ZMP continuation, with explicit finite-lambda quality.

    base_fock is a declared fixed LOCAL Hamiltonian, including any chosen
    Hartree/guide terms. coulomb_energy=0.5*sum_sigma Tr(dD_sigma J[dD_sigma])
    is in Hartree (RKS uses its spin-summed density). With grid supplied, also
    require relative grid L2<=density_tolerance. Converged SCF alone never
    certifies an inverse reference. At failure reference=None, diagnostics stay.

    correction_potential is lambda*J[dD] or 2lambda*J[dD_sigma], not XC.
    Only if hcore is supplied do we label F-hcore-J[n_recovered] as XC.
    No shift is applied to the reported physical Fock/spectrum. Inverse-map
    AD/JIT is unsupported; assessment/training against the fixed result is live.
    """
    if any(isinstance(v,jax.core.Tracer) for v in jax.tree.leaves((target,base_fock,initial,hcore))):
        raise NotImplementedError('ZMP is offline; inverse-map AD/JIT is unsupported')
    if target.layout not in ('rks','uks'):
        raise NotImplementedError('ZMP currently supports real RKS/UKS, not spinor inverse GKS')
    penalties=tuple(float(value) for value in lambdas)
    if (not penalties or any(not isfinite(v) or v<0 for v in penalties)
            or any(b<=a for a,b in zip(penalties,penalties[1:]))):
        raise ValueError('Use a finite, nonnegative, strictly increasing lambda penalty sequence')
    if not callable(coulomb):
        raise TypeError('coulomb must map a spatial AO density to its Coulomb matrix')
    if any(not isfinite(v) or v<=0 for v in (coulomb_tolerance,density_tolerance,scf_tolerance,gap_tolerance)):
        raise ValueError('ZMP tolerances must be positive and finite')
    cfg=NonlinearConfig(maxiter=maxiter,residual_tol=scf_tolerance,diis_space=diis_space,damping=damping)
    s=target.overlap; n=s.shape[0]
    numbers=(float(target.nelec)/2,) if target.layout=='rks' else tuple(float(v) for v in target.nelec)
    if any(not v.is_integer() or not 0<=v<=n for v in numbers) or sum(numbers)<=0:
        raise NotImplementedError('ZMP requires fixed integer occupied spin populations')
    counts=tuple(int(v) for v in numbers);nspin=len(counts);degeneracy=2 if nspin==1 else 1
    factor=1 if nspin==1 else 2
    base=jnp.asarray(base_fock)
    if base.shape!=target.density.shape or jnp.iscomplexobj(base):
        raise ValueError('base_fock must be real and match the target layout')
    _check(jnp.all(jnp.isfinite(base)) & jnp.allclose(base,base.swapaxes(-1,-2),atol=1e-11,rtol=1e-12),
           'base_fock must be finite and symmetric')
    _check(jnp.min(jnp.linalg.eigvalsh(s))>1e-12,'Overlap must be positive definite without clipping')
    if hcore is not None:
        hcore=jnp.asarray(hcore)
        if hcore.shape!=(n,n) or jnp.iscomplexobj(hcore):
            raise ValueError('hcore must be a real spatial AO matrix')
        _check(jnp.all(jnp.isfinite(hcore)) & jnp.allclose(hcore,hcore.T), 'hcore must be finite and symmetric')
    workspace_bytes(max_memory,resident_bytes=tree_storage_bytes(target,base,grid,hcore)+
        (20+4*diis_space)*nspin*n*n*base.dtype.itemsize)
    x=inverse_sqrt(s,1e-12)
    target_blocks=target.density[None] if nspin==1 else target.density
    base_blocks=base[None] if nspin==1 else base

    def action(density):
        matrix=jnp.asarray(coulomb(density))
        if matrix.shape!=(n,n) or jnp.iscomplexobj(matrix):
            raise ValueError('Coulomb action must return a real spatial AO matrix')
        matrix=matrix.astype(target.density.dtype)
        tolerance=1e-12+128*jnp.finfo(matrix.dtype).eps*jnp.maximum(1.,jnp.linalg.norm(matrix))
        valid=jnp.all(jnp.isfinite(matrix)) & (jnp.max(jnp.abs(matrix-matrix.T))<=tolerance)
        _check(valid,'Coulomb action must be finite and symmetric')
        return jnp.where(valid,.5*(matrix+matrix.T),jnp.nan)

    # Validate the callback eagerly, before entering compiled iterations.
    action(target_blocks[0])
    def density_from_frame(vector):
        frames=vector.reshape(nspin,n,n)
        projectors=[]
        for f,count in zip(frames,counts):
            p=(jnp.zeros_like(f) if count==0 else jnp.eye(n,dtype=f.dtype) if count==n
               else occupied_projector(.5*(f+f.T),count))
            projectors.append(degeneracy*x@p@x)
        return jnp.stack(projectors)

    def physical_fock(density,penalty):
        correction=jnp.stack([factor*penalty*action(d-target_blocks[c]) for c,d in enumerate(density)])
        return base_blocks+correction

    @jax.jit
    def physical_map(vector,penalty):
        f=physical_fock(density_from_frame(vector),penalty)
        white=x@f@x
        return (.5*(white+white.swapaxes(-1,-2))).reshape(-1)

    seed=(x@base_blocks@x).reshape(-1) if initial is None else jnp.asarray(initial).reshape(-1)
    if seed.size!=nspin*n*n or jnp.iscomplexobj(seed):
        raise ValueError('Initial orthonormal Fock must match the spin/AO dimension')
    stages=[]
    for penalty in penalties:
        solved=iterate_eager(lambda vector:physical_map(vector,penalty),seed,config=cfg)
        seed=solved.solution
        delta=density_from_frame(seed)-target_blocks
        error=float(.5*sum(jnp.sum(d*action(d).T) for d in delta))
        stages.append(ZMPStage(penalty,solved.iterations,solved.residual_norm,error,solved.converged))
        if not solved.converged:
            break

    # Resolve complete orbitals from the actual, unshifted physical Hamiltonian.
    root=s@x
    focks=root@seed.reshape(nspin,n,n)@root
    focks=.5*(focks+focks.swapaxes(-1,-2))
    eigen=[solve_hermitian(.5*(x@f@x+(x@f@x).T),config=EigenSolverConfig(method='dense',nroots=n),
        max_memory=max_memory,reserved_bytes=tree_storage_bytes(target,x,base)) for f in focks]
    coeff=jnp.stack([x@v.raw_vectors[:,:n] for v in eigen])
    energies=jnp.stack([v.raw_values[:n] for v in eigen])
    occupations=jnp.stack([degeneracy*(jnp.arange(n)<count) for count in counts])
    density=(coeff*occupations[...,None,:])@coeff.swapaxes(-1,-2)
    updated=physical_fock(density,stages[-1].penalty)
    residual=float(jnp.max(jnp.abs(x@(updated-focks)@x)))
    delta=density-target_blocks
    error=float(.5*sum(jnp.sum(d*action(d).T) for d in delta))
    gap=min((float(energies[c,count]-energies[c,count-1]) for c,count in enumerate(counts) if 0<count<n),default=float('inf'))
    xc=None if hcore is None else focks-hcore-action(density.sum(axis=0))
    if nspin==1:
        coeff,energies,occupations,density,fock=coeff[0],energies[0],occupations[0],density[0],focks[0]
        if xc is not None:xc=xc[0]
    else:
        fock=focks
    candidate=OrbitalReference(target,coeff,occupations,energies,xc_potential=xc,method='zmp')
    absolute=relative=None
    if grid is not None:
        absolute=float(jnp.sqrt(density_norm2(candidate,jnp.zeros_like(density),grid)[1]))
        norm=float(jnp.sqrt(density_norm2(candidate,-density,grid)[0]))
        relative=absolute/norm if norm>0 else float('inf')
    converged=bool(len(stages)==len(penalties) and all(v.converged for v in stages) and
        residual<=scf_tolerance and all(bool(jnp.all(v.converged)) for v in eigen))
    valid=bool(converged and gap>gap_tolerance and -coulomb_tolerance<=error<=coulomb_tolerance
        and (relative is None or relative<=density_tolerance))
    correction=updated[0]-base if nspin==1 else updated-base
    return ZMPResult(target,candidate if valid else None,density,fock,correction,tuple(stages),
        error,residual,gap,converged,valid,absolute,relative,xc)
