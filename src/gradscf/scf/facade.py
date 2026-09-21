from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

import numpy as np

from ..data.molecule import MoleculeSpec
from ..tools.jax_runtime import configure_jax_persistent_cache
from .builders import (
    build_restricted_reference_from_facade,
    build_restricted_scf_result_from_facade,
    complete_restricted_response_inputs,
    build_unrestricted_reference_from_facade,
    restricted_molecule_from_spec_with_jax_rks,
    unrestricted_molecule_from_spec_with_jax_uks,
)
from .core import _contains_jax_tracer
from gradscf.integrals.assembly import build_rks_integral_inputs
from .rks import RKSConfig, run_rks_from_integrals
from .uks import UKSConfig


def _cache_signature(value):
    """Snapshot numerical inputs by value, including arrays inside raw bases."""
    if is_dataclass(value):
        return tuple((item.name,_cache_signature(getattr(value,item.name))) for item in fields(value))
    if isinstance(value,dict):
        return tuple((repr(key),_cache_signature(item)) for key,item in sorted(value.items(),key=lambda x:repr(x[0])))
    if isinstance(value,(tuple,list)):
        return tuple(_cache_signature(item) for item in value)
    if hasattr(value,'shape') and hasattr(value,'dtype'):
        array=np.asarray(value)
        return array.dtype.str,array.shape,array.tobytes()
    return value

@dataclass
class _BaseKS:
    mol: Any
    xc: str = "pbe"
    conv_tol: float = 1e-10
    conv_tol_density: float = 1e-8
    conv_tol_grad: float = 1e-7
    max_cycle: int = 80
    damp: float = 0.0
    level_shift: float = 0.0
    grids_level: int = 0
    max_l: int = 3
    integral_backend: Literal["native", "jax", "cpu", "libcint"] = "cpu"
    geometry_grad_policy: Literal["analytic", "error", "zero"] = "analytic"
    grid_ao_backend: Literal["jax"] = "jax"
    execution_device: Literal["auto", "cpu", "gpu"] = "auto"
    init_guess: Any = "hcore"
    chkfile: str | None = None
    sap_basis: Any | None = None
    init_guess_chkfile_project: bool | None = None
    jax_compilation_cache_dir: str | None = None
    jax_persistent_cache_min_compile_time_secs: float = 0.0
    jax_persistent_cache_min_entry_size_bytes: int = 0

    e_tot: Any | None = None
    converged: bool | None = None
    mo_energy: Any | None = None
    mo_coeff: Any | None = None
    mo_occ: Any | None = None
    cycles: int | None = None
    reference: Any | None = None
    scf_result: Any | None = field(default=None,init=False,repr=False)
    _scf_inputs: Any | None = field(default=None,init=False,repr=False)
    _cached_scf_key: Any | None = field(default=None,init=False,repr=False)
    _cached_response_key: Any | None = field(default=None,init=False,repr=False)

    def run(self) -> "_BaseKS":
        self.kernel()
        return self

    def TDA(self, **kwargs: Any) -> Any:
        from .. import tdscf

        return tdscf.TDA(self, **kwargs)

    def TDDFT(self, **kwargs: Any) -> Any:
        from .. import tdscf

        return tdscf.TDDFT(self, **kwargs)

    def CIS(self, **kwargs: Any) -> Any:
        from ..ci import CIS

        return CIS(self, **kwargs)

    def CISD(self, **kwargs: Any) -> Any:
        from ..ci import CISD

        return CISD(self, **kwargs)

    def CISDT(self, **kwargs: Any) -> Any:
        from ..ci import CISDT

        return CISDT(self, **kwargs)

    def CISDTQ(self, **kwargs: Any) -> Any:
        from ..ci import CISDTQ

        return CISDTQ(self, **kwargs)

    def nuc_grad_method(self) -> "_NuclearGradient":
        return _NuclearGradient(self)

    def _spec(self) -> MoleculeSpec:
        if not hasattr(self.mol, "to_spec"):
            raise TypeError("SCF facade expects a gradscf.gto.M molecule.")
        return self.mol.to_spec()

    def _sync_from_reference(self, reference: Any) -> None:
        self.reference = reference
        self.e_tot = getattr(reference, "mf_energy", None)
        self.mo_energy = getattr(reference, "mo_energy", None)
        self.mo_coeff = getattr(reference, "mo_coeff", None)
        self.mo_occ = getattr(reference, "mo_occ", None)
        self.cycles = getattr(reference,"scf_cycles",None)
        if self.cycles is None:
            self.cycles = getattr(reference,"cycles",None)
        converged = getattr(reference,"scf_converged",None)
        self.converged = bool(getattr(reference,"converged",True) if converged is None else converged)

    def _sync_from_scf_result(self, result: Any) -> None:
        self.scf_result = result
        self.reference = None
        self.e_tot = getattr(result, "total_energy", None)
        self.mo_energy = getattr(result, "mo_energy", None)
        self.mo_coeff = getattr(result, "mo_coeff", None)
        self.mo_occ = getattr(result, "mo_occ", None)
        self.cycles = getattr(result, "cycles", None)
        self.converged = bool(getattr(result, "converged", True))

    def _configure_jax_cache(self) -> None:
        configure_jax_persistent_cache(
            cache_dir=self.jax_compilation_cache_dir,
            min_compile_time_secs=self.jax_persistent_cache_min_compile_time_secs,
            min_entry_size_bytes=self.jax_persistent_cache_min_entry_size_bytes,
        )

    def _put_on_requested_device(self, reference: Any) -> Any:
        if self.execution_device == "auto":
            return reference
        from ..tools.device import put_molecule_on_device, resolve_execution_device

        device = resolve_execution_device(self.execution_device)
        return put_molecule_on_device(reference, device=device)

    def _ensure_reference(self) -> Any:
        response_key = None
        if self._cached_scf_key is not None:
            if self._cached_scf_key != self._scf_signature():
                raise RuntimeError('Molecule or SCF configuration changed; run kernel() before preparing a reference.')
            response_key = self._response_signature()
        if self.reference is not None and (self._cached_scf_key is None or response_key == self._cached_response_key):
            return self.reference
        if self.e_tot is None:
            raise RuntimeError(
                "Run ground-state mf.kernel() or mf.run() before launching TD-SCF."
            )
        reference = self._build_reference(self._spec())
        reference = self._put_on_requested_device(reference)
        if self.scf_result is None:
            self._sync_from_reference(reference)
        else:
            # The response container stores spin-stacked orbitals. Keep the
            # facade's ground-state layout and status from the original result.
            self.reference = reference
        self._cached_response_key = response_key
        return reference


@dataclass
class RKS(_BaseKS):
    jk_backend: Literal["full", "df", "direct"] = "full"
    df_tol: float = 1e-10
    df_max_rank: int | None = None
    auxbasis: str | None = None
    direct_scf_tol: float = 0.0
    compute_local_hfx_features: bool = False
    compute_local_hfx_aux: bool = False
    hfx_omega_values: tuple[float, ...] = (0.0, 0.4)
    hfx_chunk_size: int = 512

    def _config(self) -> RKSConfig:
        return RKSConfig(
            xc_spec=self.xc,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
            conv_tol_density=self.conv_tol_density,
            conv_tol_grad=self.conv_tol_grad,
            damping=self.damp,
            level_shift=self.level_shift,
            jk_backend=self.jk_backend,
            df_tol=self.df_tol,
            df_max_rank=self.df_max_rank,
            auxbasis=self.auxbasis,
            direct_scf_tol=self.direct_scf_tol,
        )

    def _build_reference(self, spec: MoleculeSpec) -> Any:
        if self._scf_inputs is not None:
            self._scf_inputs = complete_restricted_response_inputs(self._scf_inputs,spec,self.mol.basis)
        return build_restricted_reference_from_facade(
            spec,
            **self._build_options(spec),
            compute_local_hfx_features=self.compute_local_hfx_features,
            compute_local_hfx_aux=self.compute_local_hfx_aux,
            hfx_omega_values=self.hfx_omega_values,
            hfx_chunk_size=self.hfx_chunk_size,
            include_dipole_integrals=True,
            reference_builder=restricted_molecule_from_spec_with_jax_rks,
            scf_inputs=self._scf_inputs,
            scf_result=self.scf_result,
        )

    def _build_scf_result(self, spec: MoleculeSpec) -> Any:
        result, inputs = build_restricted_scf_result_from_facade(
            spec,
            **self._build_options(spec),
            build_inputs_fn=build_rks_integral_inputs,
            run_rks_fn=run_rks_from_integrals,
            return_inputs=True,
        )
        self._scf_inputs = inputs
        return result

    def _build_options(self, spec):
        return dict(mol=self.mol,xc=self.xc,grids_level=self.grids_level,max_l=self.max_l,
            integral_backend=self.integral_backend,geometry_grad_policy=self.geometry_grad_policy,
            grid_ao_backend=self.grid_ao_backend,rks_config=self._config(),init_guess=self.init_guess,
            chkfile=self.chkfile,sap_basis=self.sap_basis,init_guess_chkfile_project=self.init_guess_chkfile_project,
            geometry_is_traced=_contains_jax_tracer(spec))

    def _scf_signature(self):
        return _cache_signature((self._spec(),self.mol.basis,self._config(),self.grids_level,
            self.max_l,self.integral_backend,self.geometry_grad_policy,self.grid_ao_backend,
            self.init_guess,self.chkfile,self.sap_basis,self.init_guess_chkfile_project))

    def _response_signature(self):
        return _cache_signature((self.compute_local_hfx_features,self.compute_local_hfx_aux,
            self.hfx_omega_values,self.hfx_chunk_size,self.execution_device))

    def kernel(self) -> Any:
        self._configure_jax_cache()
        for name in ('reference','scf_result','_scf_inputs','_cached_scf_key','_cached_response_key',
                     'e_tot','mo_energy','mo_coeff','mo_occ','cycles','converged'):
            setattr(self,name,None)
        result = self._build_scf_result(self._spec())
        self._sync_from_scf_result(result)
        self._cached_scf_key = self._scf_signature()
        return self.e_tot

    def density_fit(self, auxbasis: str | None = None) -> "RKS":
        self.jk_backend = "df"
        self.auxbasis = auxbasis or 'def2-universal-jkfit'
        return self

    def direct_scf(self) -> "RKS":
        self.jk_backend = "direct"
        return self


@dataclass
class UKS(_BaseKS):
    conv_tol_grad: float = 1e-7
    jk_backend: Literal['full','df'] = 'full'
    auxbasis: str | None = None

    def _config(self) -> UKSConfig:
        return UKSConfig(
            xc_spec=self.xc,
            max_cycle=self.max_cycle,
            conv_tol=self.conv_tol,
            conv_tol_density=self.conv_tol_density,
            conv_tol_grad=self.conv_tol_grad,
            damping=self.damp,
            level_shift=self.level_shift,
            jk_backend=self.jk_backend,
            auxbasis=self.auxbasis,
        )

    def _build_reference(self, spec: MoleculeSpec) -> Any:
        return build_unrestricted_reference_from_facade(
            spec,
            mol=self.mol,
            xc=self.xc,
            grids_level=self.grids_level,
            max_l=self.max_l,
            integral_backend=self.integral_backend,
            geometry_grad_policy=self.geometry_grad_policy,
            grid_ao_backend=self.grid_ao_backend,
            uks_config=self._config(),
            init_guess=self.init_guess,
            chkfile=self.chkfile,
            sap_basis=self.sap_basis,
            init_guess_chkfile_project=self.init_guess_chkfile_project,
            reference_builder=unrestricted_molecule_from_spec_with_jax_uks,
        )

    def kernel(self) -> Any:
        self._configure_jax_cache()
        reference = self._build_reference(self._spec())
        reference = self._put_on_requested_device(reference)
        self._sync_from_reference(reference)
        return self.e_tot

    def density_fit(self, auxbasis: str | None = None) -> "UKS":
        self.jk_backend='df'
        self.auxbasis=auxbasis or 'def2-universal-jkfit'
        return self

    def direct_scf(self) -> "UKS":
        raise NotImplementedError("UKS direct SCF is not exposed by the current core solver.")


@dataclass(frozen=True)
class _NuclearGradient:
    mf: _BaseKS

    def kernel(self) -> Any:
        raise NotImplementedError(
            "Explicit SCF coordinate differentiation is disabled. "
            "Use implicit-differential training workflows instead."
        )
