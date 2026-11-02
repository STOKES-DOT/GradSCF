# NNAO primitive-pool and valence-zeta comparison

Nine explicit families combine S/M/L primitive pools with SZP/DZP/TZP.
The existing CHNOF campaign below retains its original inputs. Expanded main-group
vocabularies and reference-core policy are described in [ELEMENTS.md](ELEMENTS.md).
All use spherical functions, fixed single-zeta core contractions, all electrons,
and one polarization shell. Current reference cores use original 6-31G s/p6 and
Ga-Kr d3 contractions. DZ/TZ multiply only the valence radial contractions.
The numerical tables below describe the original CHNOF subset.

| Tier | C/N/O/F active primitives | H primitives | Mixed-pool AO: heavy / H |
| --- | --- | --- | --- |
| S | 3s3p2d | 3s2p | 23 / 9 |
| M | 4s4p2d | 4s2p | 27 / 10 |
| L | 6s6p3d | 6s2p | 40 / 12 |

| Zeta | Contracted heavy atom | Contracted H | AO: heavy / H |
| --- | --- | --- | --- |
| SZP | core 1s + valence 1s1p + 1d | 1s1p | 10 / 4 |
| DZP | core 1s + valence 2s2p + 1d | 2s1p | 14 / 5 |
| TZP | core 1s + valence 3s3p + 1d | 3s1p | 18 / 6 |

Names are `nnao_{szp,dzp,tzp}_{s,m,l}`. L-SZP preserves the existing
`szp663_direct` exponents, contractions, and AO ordering. M-H uses 4s2p,
so M is not an alias for the older `szp442_direct` H template.

Each active shell binds an nprimitive-by-ncontraction coefficient matrix.
MACE outputs separate complete radial columns, shared across angular m components.
Extra valence columns start from diffuse-first primitive directions orthogonalized
in the normalized Gaussian overlap metric. This supplies independent columns;
it is not an atomic-SCF reference fit. Core parameters are unchanged.

For S-TZP, three contractions span the entire three-primitive valence space.
Nonsingular transformations of these valence columns cannot improve a converged
SCF energy. Its remaining coefficient flexibility is in polarization channels.

## Sequential workflow

`workflows.basis_grid run` keeps native A/B integrals and exponent VJPs on CPU,
and runs their SCF/implicit response and free-contraction optimization on GPU.
Use `--calibration-platform cpu` for small CPU reference checks. Each stage
runs in a separate process; fixed-exponent MACE stage C remains on GPU.
Small reference problems are H2, CH4, NH3,
H2O, HF and CO, neutral singlet RHF with explicit unit weights and
def2-universal-jkfit. Default budgets are 20 exponent steps, 300 free-contraction
steps and three starts. Selection uses the lowest weighted mean free-contraction
energy among complete valid candidates. Finite-budget results and rejected steps
are recorded; selection is not evidence of global convergence.

L-family runs can reuse the existing accepted exponent candidate using
`--fixed-candidate`; they still perform B for their own contraction multiplicity
and recheck the exponent direction. S/M are separately calibrated at their exact
primitive counts. The selected exponent pools may differ across zeta families;
comparison is between independently calibrated bases, not a strict nested-space
variational hierarchy.

Stage C uses the same 200 training structures, seed 0, fresh Adam at 1e-4,
1000 epochs and no validation split. Fixed S/h/packed DF values use mmap caches;
full four-index ERIs are never stored. Cold/hot consistency and all-molecule
parameter FD checks gate every training start or resume. Trainable masks exclude
MACE's immutable activation constants. Invalid candidates never replace accepted
parameters/Adam. Stage D is not launched by this workflow.

Campaign definitions, source hashes, references and candidate selection are
recorded separately for each family. An incomplete A/B directory fails explicitly
instead of being overwritten; stage C can resume its accepted checkpoint under
the identical source/configuration identity.

## Historical 200-system resource plan

The plan below describes the original campaign. Its eight additional jobs were
later archived/cancelled; only original job 96510 was retained. Expanded training
is paused pending the revised single-zeta-core calibration and validation.

Three serial Slurm lanes: SZP on the existing n004 A800 allocation lineage;
DZP and TZP each request one A100_80G. Existing L-SZP job 96510 is retained as
the ninth comparison, not restarted. New SZP S/M jobs depend on its successful
completion. Jobs within a lane have `afterok` dependencies, and all new jobs
depend on the nine-family validation job. Each new training job requests one
GPU, 32 CPUs and 256 GiB; total simultaneous CPU allocation is at most 96,
following the user's updated per-job CPU requirement. Slurm determines the physical GPU after allocation;
the SZP lane requests the same node/model, not a guaranteed GPU UUID.

New runs use the independent `GradSCF_basis_grid_20261013` source directory.
The currently running source tree and existing caches/checkpoints are untouched.
