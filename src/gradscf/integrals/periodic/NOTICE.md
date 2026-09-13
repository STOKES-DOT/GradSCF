# GTH data provenance

`gth_data.json` contains basis and pseudopotential parameters exported from the
PySCF 2.9.0 distribution (`pyscf.pbc.gto.basis` and `pyscf.pbc.gto.pseudo`).
The included named families are gth-szv, gth-dzvp, gth-tzvp and gth-pade,
gth-pbe, gth-blyp. Raw basis/parameter dictionaries are also accepted.

PySCF is copyright The PySCF Developers and distributed under Apache License
2.0. The full license is bundled at
`gradscf/integrals/_native/vendor/pyscf/LICENSE` in this distribution.
Upstream source: https://github.com/pyscf/pyscf/tree/v2.9.0/pyscf/pbc/gto

Numerical algorithms in this directory implement analytic Cartesian Gaussian
Fourier transforms, normalized GTH projectors, Ewald summation and FFT Coulomb
convolution. No PySCF Python API is imported at runtime.
