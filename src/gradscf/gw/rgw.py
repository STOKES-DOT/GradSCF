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
from .g0w0 import g0w0_cd_restricted


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
        result = getattr(mf, "scf_result", None)
        if result is None:
            raise RuntimeError(
                "Restricted GW expects the RKS facade result (mf.scf_result); "
                "run mf.kernel() first."
            )
        mo_occ = np.asarray(result.mo_occ)
        nocc = int(np.count_nonzero(mo_occ > 0.0))
        res = g0w0_cd_restricted(
            mo_energy=result.mo_energy,
            mo_coeff=result.mo_coeff,
            nocc=nocc,
            df_factors=self._df_factors(),
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
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "GW":
        self.kernel(orbs)
        return self


__all__ = ["GW"]
