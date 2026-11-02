# Fixed STO-6G polarization controls for ethane torsion

User-authorized comparison on the existing 25 rigid ethane geometries, 0–120°
in 5° steps. Three fixed bases are added: STO-6G (16 AO), STO-6G plus one carbon
d shell (26 AO), and STO-6G plus carbon d and hydrogen p (44 AO). Spherical d
has five components. These are explicit custom additions; the labels do not
assert that a unique standardized STO-6G* or STO-6G** parameterization exists.

STO-6G primitives and coefficients come unchanged from the installed PySCF
library. The carbon d exponent 0.8 and hydrogen p exponent 1.1 Bohr^-2, with
raw coefficients 1.0, are copied from the archived 6-31G(d,p) basis used in
the original benchmark. All added shells remain fixed during torsion.

DF-RHF/def2-universal-jkfit, float64, spherical AO, SCF energy tolerance 1e-12 Eh,
orbital-gradient tolerance 1e-9, maximum 200 cycles; final AO commutator residual
must be below 1e-7. The zero-shift solution is required after any rescue.
GPU4PySCF is used only in this comparison artifact, with the same protocol as
the existing conventional-basis results. No production code or model is changed.

NNAO 44-AO curves reuse frozen MACE update 3351 and same-pool free contraction;
the additional 50-AO second-carbon-p free curve is identified separately.
No MACE checkpoint is silently replaced by the later epoch-596 model.

Equal contracted AO count controls the SCF matrix dimension. Primitive counts,
exponents, core contractions, and contraction adaptation differ, so this is not
an isolated measurement of the MACE contribution. Report absolute energies and
relative torsion curves separately; one favorable barrier does not establish
broad basis accuracy. HF/cc-pVTZ is a comparison basis, not an exact reference.

Resources: one V100, 32 CPU, 64 GiB requested memory, at most 10 minutes.
Submit from the frozen remote checkout:
`sbatch artifacts/nnao-ethane-sto6g-polarized-20261020/run.slurm`.
