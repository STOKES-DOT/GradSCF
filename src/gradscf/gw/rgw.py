"""PySCF-style restricted GW facade.

Example
-------
>>> from gradscf import gto, dft
>>> from gradscf.gw import GW
>>> mol = gto.M(atom="O 0 0 0.117; H 0 0.755 -0.471; H 0 -0.755 -0.471",
...             basis="cc-pvdz")
>>> mf = dft.RKS(mol, xc="hf").run()
>>> gw = GW(mf).run()
>>> gw.mo_energy  # quasiparticle energies

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..df import eri_pair_matrix_to_df_factors
from .g0w0 import g0w0_cd_restricted, _mo_factors
from ..scf.reference import reference_state_signature, _array_signature


class GW:
    """Spin-restricted G0W0 (contour deformation) on top of RHF/RKS.

    Parameters
    ----------
    mf:
        A converged :class:`gradscf.dft.RKS` facade object (any ``xc``,
        including ``"hf"``).
    nw:
        Imaginary-axis quadrature size (default 100, PySCF convention).
    eta:
        Broadening of the Green's function / retarded response (1e-3).
    """

    def __init__(self, mf, *, nw: int = 100, eta: float = 1e-3):
        if getattr(mf, "mo_energy", None) is None:
            raise RuntimeError("GW requires a converged mean-field object; call mf.kernel() first.")
        self._scf = mf
        self.nw = int(nw)
        self.eta = float(eta)
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = getattr(mf, "mo_occ", None)
        self.converged = None
        self.result = None
        self._source_key = None
        self._ao_factors = None
        self._dipole_ao = None

    def _source_signature(self):
        mf = self._scf
        result = getattr(mf, "scf_result", None)
        inputs = getattr(mf, "_scf_inputs", None)
        if result is None or inputs is None or not mf.converged:
            raise RuntimeError("Run and converge the restricted SCF before GW")
        if mf._cached_scf_key != mf._scf_signature():
            raise RuntimeError("SCF inputs changed; run SCF and GW again")
        arrays = [
            result.mo_energy,
            result.mo_coeff,
            result.mo_occ,
            result.fock_matrix,
            result.hcore_matrix,
            result.density_matrix,
            getattr(inputs, "df_factors", None),
            getattr(inputs, "eri_pair_matrix", None),
        ]
        for name in ("mo_energy", "mo_coeff", "mo_occ"):
            if _array_signature(getattr(mf, name)) != _array_signature(
                getattr(result, name)
            ):
                raise RuntimeError("SCF orbitals changed; run SCF and GW again")
        return (
            self.nw,
            self.eta,
            reference_state_signature(mf),
            tuple(None if x is None else _array_signature(x) for x in arrays),
        )

    def state_signature(self):
        """Checked eager provenance for post-GW consumers."""
        if self.result is None or self._source_key is None:
            raise RuntimeError("Run GW before requesting its reference data")
        if self._source_signature() != self._source_key:
            raise RuntimeError("GW source or settings changed; run GW again")
        if _array_signature(self.mo_energy) != _array_signature(
            self.result.mo_energy
        ) or _array_signature(self.mo_coeff) != _array_signature(self.result.mo_coeff):
            raise RuntimeError("GW orbitals changed; run GW again")
        dipole = getattr(self._scf._scf_inputs, "dipole_integrals", None)
        if dipole is None:
            dipole = self._dipole_ao
        arrays = (
            self.result.mo_energy,
            self.result.mo_coeff,
            self.result.qp_computed_mask,
            self.result.converged_mask,
            self._ao_factors,
            dipole,
        )
        return (
            self._source_key,
            id(self.result),
            tuple(None if x is None else _array_signature(x) for x in arrays),
        )

    def get_bse_inputs(self, *, max_aux=1024, max_factor_elements=20_000_000):
        """Validated G0W0/W0 snapshot; no BSE implementation is imported here."""
        if self.result is None or self._ao_factors is None:
            raise RuntimeError("Run GW before requesting BSE inputs")
        naux = self._ao_factors.shape[0]
        nmo = self.result.mo_coeff.shape[1]
        if naux > max_aux:
            raise ValueError("BSE screening exceeds max_aux before MO transformation")
        if naux * nmo * nmo > max_factor_elements:
            raise ValueError(
                "BSE factors exceed max_factor_elements before MO transformation"
            )
        self.state_signature()
        mf = self._scf
        if self.result.qp_computed_mask is None or self.result.converged_mask is None:
            raise ValueError("GW result does not contain QP coverage/convergence metadata")
        available = getattr(mf._scf_inputs, "dipole_integrals", None)
        if available is not None:
            self._dipole_ao = available
        elif self._dipole_ao is None:
            from ..scf.builders import complete_restricted_response_inputs

            completed = complete_restricted_response_inputs(
                mf._scf_inputs, mf._spec(), mf.mol.basis, cart=mf.mol.cart
            )
            self._dipole_ao = completed.dipole_integrals
        return dict(
            qp_energy=self.result.mo_energy,
            screening_energy=mf.scf_result.mo_energy,
            mo_factors=_mo_factors(self._ao_factors, self.result.mo_coeff),
            mo_coeff=self.result.mo_coeff,
            ao_factors=self._ao_factors,
            nocc=int(np.count_nonzero(np.asarray(mf.mo_occ))),
            dipole_ao=self._dipole_ao,
            qp_computed_mask=self.result.qp_computed_mask,
            qp_converged_mask=self.result.converged_mask,
        )

    def _df_factors(self):
        inputs = getattr(self._scf, "_scf_inputs", None)
        factors = getattr(inputs, "df_factors", None) if inputs is not None else None
        if factors is not None:
            return factors
        pair = getattr(inputs, "eri_pair_matrix", None) if inputs is not None else None
        if pair is None:
            raise RuntimeError(
                "GW requires low-rank ERI factors or the packed ERI pair "
                "matrix on the mean-field object; neither was found."
            )
        nao = int(np.asarray(self._scf.mo_coeff).shape[0])
        return eri_pair_matrix_to_df_factors(pair, nao=nao, tol=1e-12)

    def kernel(self, orbs: Sequence[int] | None = None):
        mf = self._scf
        signature=self._source_signature()
        result = getattr(mf, "scf_result", None)
        if result is None:
            raise RuntimeError(
                "Restricted GW expects the RKS facade result (mf.scf_result); "
                "run mf.kernel() first."
            )
        mo_occ = np.asarray(result.mo_occ)
        nocc = int(np.count_nonzero(mo_occ > 0.0))
        if result.mo_coeff.ndim != 2 or np.iscomplexobj(result.mo_coeff):
            raise NotImplementedError("Restricted GW facade requires real closed-shell orbitals")
        if not np.array_equal(mo_occ,np.r_[np.full(nocc,2.),np.zeros(len(mo_occ)-nocc)]):
            raise ValueError("Restricted GW requires integer closed-shell occupations, occupied first")
        factors=self._df_factors()
        res = g0w0_cd_restricted(
            mo_energy=result.mo_energy,
            mo_coeff=result.mo_coeff,
            nocc=nocc,
            df_factors=factors,
            fock_matrix=result.fock_matrix,
            hcore_matrix=result.hcore_matrix,
            density_matrix=result.density_matrix,
            nw=self.nw,
            eta=self.eta,
            orbs=orbs,
        )
        self.result = res
        self.converged = res.converged
        self.mo_energy = res.mo_energy
        self.mo_coeff = res.mo_coeff
        self._ao_factors = factors
        self._source_key = signature
        self._dipole_ao = None
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "GW":
        self.kernel(orbs)
        return self


__all__ = ["GW"]
