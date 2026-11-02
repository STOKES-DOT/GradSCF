# NNAO core basis benchmarks

Three user-requested measurements, with all-electron traditional comparisons:

1. Atomic radial electron-number distributions of H, H-, C, C-. Explicit
   charge/spin values are (0,1), (-1,0), (0,2), (-1,3); UHF for nonzero spin,
   RHF for spin zero. Integrate P(r) to the electron number and also show P/Ne.
2. S22 Boys-Bernardi orbital BSSE on all 22 dimers. All five energies retain
   frozen dimer geometry and one common dimer auxiliary metric. For MACE, all
   orbital contractions are frozen from the dimer prediction, including ghost
   functions. Ghost atoms are never passed as chemical neighbours to MACE.
   Monomer files in GMTKN55 can be relaxed; their connectivity validates the
   two fragments, while the CP calculations use coordinates cut from the dimer.
3. Static isotropic polarizabilities of benzene, naphthalene, anthracene,
   p-nitroaniline, cyclohexane, hexane, ethylene, butadiene, hexatriene. Use
   zero-field SCF linear response, not finite fields or finite differences.

This first implementation uses DF-HF/def2-universal-jkfit throughout, float64,
spherical AOs. It is a new HF comparison, not numerical reproduction of the
paper's omegaB97X-D4 results. Polarizability geometries are independently
sourced PubChem 3D conformers, not claimed to be the paper's coordinates.
All raw SDF files and data SHA256 values are preserved.

Traditional bases: STO-6G, STO-6G+P (original def2-SVP polarization),
6-31G(d,p), def2-SVP, cc-pVDZ, cc-pVTZ, ma-def2-QZVPP. Compare frozen MACE
update 3351 without switching checkpoints. Additional aug-cc-pV5Z atomic
references replace neither the method nor the paper's explicitly different
uncontracted 14s example; their identity is recorded.

The MACE model receives explicit total charge and spin. Atomic charge/spin
states are extrapolations from its neutral closed-shell molecular training,
and are labelled accordingly. Its inputs contain no electric field: report
fixed-predicted-basis SCF polarizabilities only. Field-adaptive basis response
is not implemented or inferred from these measurements.

SCF tolerance 1e-12 Eh, density 1e-10, orbital gradient 1e-9, AO commutator
residual <=1e-7; numerical linear dependence threshold 1e-10. Response uses
the existing DFPT residual and implicit linear solver (rtol1e-10/atol1e-12),
checks tensor reciprocity and particle conservation. This checks a stationary
HF branch; it is not proof of its global optimality or full orbital stability.
Full ERIs are not requested. Packed DF factors are unpacked only to three
indices at the existing SCF boundary. Native library and original production
source stay unchanged by the comparison runners.

Local gates: protocol math, radial normalization, p-orientation invariance,
CP signs/nested spaces/packed ordering, native H and H2, native He2 CP,
and independent PySCF CPHF on H2, UHF C/C-, and all five He2 CP energies
using the same dimer auxiliary basis. Sixteen local checks passed.
No finite-difference checks.

Smoke profile: H/H-, S22 water dimer, ethylene; STO-6G+P, def2-SVP and MACE
(12 cases). One V100, 32 CPU, 128 GiB, maximum 20 minutes.

The remote smoke test completed 12/12 cases (job 101883, V100); its numerical
loop took 67.36 seconds, excluding startup/imports and checkpoint loading.

Full profile approved by the user on 2026-10-20: 284 cases (36 atomic,
176 CP sets, 72 polarizabilities), about 988 primal SCF solves and 216 field
linear solves. One A100 80G, 32 CPU, 512 GiB, maximum 3 hours. Each accepted
case is saved; failures remain explicit and are not replaced by other methods.

Prepare data:
`python tests/comparisons/nnao_core/prepare.py --output artifacts/nnao-core-benchmarks-20261020`.
Run gates:
`JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q tests/nnao/test_core_benchmark_protocols.py tests/nnao/test_core_benchmark_native.py`.
Run the approved profile from the frozen remote checkout with the matching
Slurm script. Benchmark modules live in tests/comparisons, not production.
