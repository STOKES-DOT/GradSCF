# Frozen 27-basis GPU4PySCF accuracy/size benchmark

Approved on 2026-10-29: 18 conventional bases + nine final NNAO models;
30 fixed neutral, closed-shell H/C/N/O structures; DF-MP2 and omegaB97M-V.
The model basis is predicted once per physical structure and reused by both
methods. No free contraction, retraining or finite-difference gradients.

## Data and scoring

`inputs/structures.json` contains ISO34/3,14,15,24,25,26 reactants/products,
and S22/1,2,8,9,10,16 dimers plus fragments extracted from their actual
coordinates. The official GMTKN files interleave atoms by element; bonded
components identify fragments. Reference S22 dissociation signs are reversed
for the negative interaction-energy convention. Source hashes are frozen.

Methane monomer source S22/08a occurs in training. This is labelled separately
from the stricter distance-spectrum geometry comparison (no matches at 1e-7 A).
No claim that the entire benchmark is chemically unseen is made.

Main score: equal weight for six isomerization errors and six uncorrected
interaction-energy errors, kcal/mol. Fixed-model isolated fragment references
are used. Ground-state errors use the same method/aug-cc-pVQZ finite reference,
not CBS. This campaign does not compute CP, BSSE, ghost monomers, or
monomers with dimer-conditioned coefficients. Dimer interactions are the
uncorrected E(AB)-E(A)-E(B), using each physical structure's frozen basis.
Failed physical cases are not silently removed from scores.

## Numerical settings

PySCF 2.8.0, GPU4PySCF-CUDA12 1.7.0, FP64, spherical AOs. RHF orbitals for
DF-MP2, freeze one 1s orbital per real C/N/O, `with_t2=False`. J/K auxiliary
def2-universal-jkfit; MP2 auxiliary aug-cc-pVQZ-ri, explicitly frozen in the input files.
RKS omegaB97M-V includes VV10. Both `grids.level` and `nlcgrids.level` are 3.
SCF energy 1e-10 Eh, orbital-gradient tolerance 1e-7, max 200 cycles.

Export checks each checkpoint/source SHA and the primitive/fixed-core layout
against training cache metadata. Export runs on CPU in independent model
processes; GPU4PySCF does not share an allocator with JAX.

## Size and resources

AO mean is over the same 30 physical structures.
Bubble area is proportional to the mean number of unique spherical primitive
functions, including fixed cores and angular degeneracy. Shared contraction
columns do not duplicate primitives. Actual run times and memory are separate.

Only 1620 method-energy cases + 60 finite-reference cases are requested,
1680 in total. An MP2 energy case includes its RHF solve. No CP/BSSE cases
are part of execution or scoring.

CPU export already completed. Formal lanes: V100/16 CPU/64 GiB/8 h
(small/SZP/DZP), A800/16 CPU/192 GiB/8 h (TZ/TZP),
H200/32 CPU/256 GiB/16 h (QZ/reference). Effective CUDA 12.8 comes from
the cluster NVHPC 25.3 Toolkit; Python packages use gpu4pyscf_bench.
Benchmark GPU cap three, account CPU cap 64; unrelated work is preserved.
Per the user's latest instruction, run the frozen benchmark directly, with
no CPU/GPU comparison or numerical preflight as a release condition. This
policy is recorded as skipped_by_explicit_user_instruction, not a passed
preflight. SCF convergence and per-case validity checks remain part of the
requested calculation. The physical-only workflow reuses the unchanged
run_gpu_engine.py numerical source and existing energy fingerprints, so
removing CP does not invalidate finished physical results. No mixed-device
time ranking.

## Outputs

Raw energies, explicit frozen bases, checkpoint/source lineage, counts,
timings, memory and missing-case statuses remain under this permanent remote
campaign directory. Local handoff goes to NNAO_paper data only, with CSV,
arrays, PNG/PDF/SVG and reproduction scripts; manuscript text is unchanged.
