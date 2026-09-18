"""PySCF-style unrestricted GW facade.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import jax.numpy as jnp

from ..df import eri_pair_matrix_to_df_factors
from .g0w0 import g0w0_cd_unrestricted


class UGW:
    """Spin-unrestricted G0W0 (contour deformation) on top of UHF/UKS.

    Parameters
    ----------
    mf:
        A converged :class:`gradscf.dft.UKS` facade object.  The
        unrestricted reference (``mf.reference``) supplies spin-resolved
        orbital energies/coefficients, densities and ``h1e``.
    nw, eta:
        Imaginary-grid size and broadening.
    """

    def __init__(self, mf, *, nw: int = 100, eta: float = 1e-3):
        if getattr(mf, "mo_energy", None) is None:
            raise RuntimeError("UGW requires a converged mean-field object; call mf.kernel() first.")
        self._scf = mf
        self.nw = int(nw)
        self.eta = float(eta)
        self.mo_energy = None
        self.mo_coeff = getattr(mf, "mo_coeff", None)
        self.mo_occ = getattr(mf, "mo_occ", None)
        self.converged = None
        self.result = None

    def _df_factors(self, reference):
        factors = getattr(reference, "df_factors", None)
        if factors is not None:
            return factors
        inputs = getattr(self._scf, "_scf_inputs", None)
        pair = getattr(inputs, "eri_pair_matrix", None) if inputs is not None else None
        if pair is not None:
            nao = int(np.asarray(self._scf.mo_coeff)[0].shape[0])
            return eri_pair_matrix_to_df_factors(pair, nao=nao, tol=1e-12)
        # Last resort: rebuild the packed ERI from the molecular spec (the
        # UKS facade does not cache integral inputs).
        from ..integrals import basis_from_pyscf_spec, eri_pair_matrix_packed

        mol = self._scf.mol
        basis = basis_from_pyscf_spec(
            mol.atom,
            basis=mol.basis,
            unit=mol.unit,
            charge=mol.charge,
            spin=mol.spin,
            cart=mol.cart,
        )
        pair = eri_pair_matrix_packed(basis)
        return eri_pair_matrix_to_df_factors(pair, nao=basis.nao, tol=1e-12)

    def kernel(self, orbs: Sequence[int] | None = None):
        mf = self._scf
        reference = getattr(mf, "reference", None)
        if reference is None:
            raise RuntimeError(
                "Unrestricted GW expects the UKS unrestricted reference "
                "(mf.reference); run mf.kernel() first."
            )
        mo_energy = np.asarray(reference.mo_energy)
        mo_coeff = np.asarray(reference.mo_coeff)
        mo_occ = np.asarray(reference.mo_occ)
        if mo_energy.shape[0] != 2:
            raise RuntimeError("mf.reference does not look unrestricted (expected spin axis 0 of size 2).")
        nocc_a = int(np.count_nonzero(mo_occ[0] > 0.0))
        nocc_b = int(np.count_nonzero(mo_occ[1] > 0.0))
        rdm1 = np.asarray(reference.rdm1)
        df_factors = self._df_factors(reference)

        # The unrestricted reference does not carry Fock matrices.  For an
        # HF starting point they are reconstructed from the densities inside
        # the driver; DFT starting points need the UKSResult Fock matrices,
        # which are not exposed by the current facade -- fail explicitly.
        hfx = float(getattr(reference, "exact_exchange_fraction", 0.0) or 0.0)
        if abs(hfx - 1.0) > 1e-12:
            raise NotImplementedError(
                "Unrestricted GW currently supports HF starting points only "
                "(v^mf is rebuilt as -K_sigma).  DFT starting points require "
                "spin Fock matrices that the UKS facade does not expose yet."
            )
        res = g0w0_cd_unrestricted(
            mo_energy=(jnp.asarray(mo_energy[0]), jnp.asarray(mo_energy[1])),
            mo_coeff=(jnp.asarray(mo_coeff[0]), jnp.asarray(mo_coeff[1])),
            nocc=(nocc_a, nocc_b),
            df_factors=df_factors,
            fock_matrix=None,
            hcore_matrix=None,
            density_matrix=(jnp.asarray(rdm1[0]), jnp.asarray(rdm1[1])),
            nw=self.nw,
            eta=self.eta,
            orbs=orbs,
        )
        self.result = res
        self.converged = res.converged
        self.mo_energy = res.mo_energy
        self.mo_coeff = res.mo_coeff
        return self.mo_energy

    def run(self, orbs: Sequence[int] | None = None) -> "UGW":
        self.kernel(orbs)
        return self


__all__ = ["UGW"]
