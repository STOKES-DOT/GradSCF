# SCF-100: fixed TDD and first implementation checkpoint

This is a checkpoint, not completion of the large-system performance matrix.
The acceptance contract is [TDD_PLAN.md](TDD_PLAN.md), its byte fingerprints
are in [TDD_LOCK.json](TDD_LOCK.json), and progress is tracked separately in
[TDD_STATUS.json](TDD_STATUS.json).

## Scope and provenance

The parent is `40a569bf9c9515e92e30d499f3784266ad2408a8`. Work is isolated on
`perf/scf-100-grid3-20261105` in `/Users/jiaoyuan/.codex/worktrees/5fbb/GradSCF`.
The previous 60-system controller and its active scientific step were stopped
on explicit user instruction; their raw results were retained. Allocation
115927 was retained for the new task, not an additional GPU allocation.

The sole large molecule is the existing beta-carotene conformer, C40H56,
96 atoms, 296 electrons, Cartesian def2-SVP, KS grid level 3. Its measured basis
dimension is 880 AO. The canonical plan-v2 contains 30 method/representation
configurations per engine: three HF methods and three KS methods with four
functionals, each with explicit auxiliary RI or conventional direct J/K.

All scientific validation runs on jy_gpu_2/n006 under Slurm, CPU float64.
Narrow regressions use eight allocated logical CPUs; native public workflows
and the large case use 64. The existing Python 3.11 comparison environment and
unchanged libcint extension are reused. Its SHA256 is
`ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.
Actual tested Python source fingerprints are in [candidate_source.json](candidate_source.json).
No local scientific calculation or environment installation was performed.

## Fixed-contract evidence

The original source produced 29 expected RED cases and four passing existing
full/packed contracts in 13.46 seconds. The first consumer candidate then
passed all 33 contracts in 25.30 seconds. A separate six-case compatibility
gate caught the empty-HF-grid problem before large execution. Lazy grid
completion preserves spatial integrals and SCF state while restoring neural
XC and HFX reference consumers.

The combined regressions initially passed 78 cases and exposed one incomplete
test result fixture. Correcting its required HF orbital fields, without
changing assertions or tolerances, passed that case. Two real external-XC
public workflows also passed. The existing implicit and explicit force-loss
mixed-derivative tests passed in 22.62 seconds after their unchanged example
dependency was copied into the isolated test tree.

There are **83 distinct related regression cases with final passing results**
across these runs, plus seven metadata-only tests. Groups overlap and must not
be summed. [validation_summary.json](validation_summary.json) preserves group
counts, errors, timings and original JUnit hashes. Fixture corrections and the
initial failed records are documented in [TDD_AMENDMENTS.md](TDD_AMENDMENTS.md).

## First production changes

- UKS RI exchange uses occupied widths per spin, with physical density fallback
  for inconsistent orbital hints, fractional tails and non-prefix occupations.
- UKS direct products are carried in SCF state, updated from the actual density
  increment and freshly rebuilt before convergence certification.
- GHF/GKS consume true RI factors or direct spatial operators while retaining
  all complex spin blocks. GKS also consumes bounded grid blocks.
- RSH RI is explicitly RI-J/direct-K through `only_dfj=True`; SR/LR exchange
  remains exact within the registered integral screening. It is not RI-JK.
- HF energy preparation can omit quadrature; grid-consuming references lazily
  complete it without replacing integrals or repeating SCF.

No new stop-gradient boundary, orbital truncation, reduced grid, alternative
Gaussian integral engine or relaxed convergence gate was introduced.

## Real native public-entry smoke

Nine independent methane/STO-3G workflows passed their scientific gates:
six GradSCF workflows and three PySCF RSH references. These exercise native
GHF RI/direct, RKS/UKS/GKS CAM-B3LYP RI-J/direct-K, and UKS/PBE0 direct with
fresh-object repeats. [native_smoke.json](native_smoke.json) retains results.

The RKS/UKS/GKS CAM-B3LYP energies differ from their respective PySCF values
by at most approximately 1.21e-11 Ha. PySCF retained packed exact exchange for
these tiny references, so their timing status is not_comparable to direct-K.
This is a physical value check, not a cross-engine speed claim.

## Large baseline and remaining gates

The first 96-atom original-source RKS/PBE/RI baseline has started in the
separate `baseline/` source tree. Its module path and 880-AO dimension were
verified; no converged result or performance ratio is available at this
checkpoint. It has a 200 GiB group RSS guard and 6000 seconds per SCF call.

The complete large matrix, matched parent/candidate performance, remaining
NNAO/two-step derivative gates and budget tuning are still open. Small-array
or small-native success is not large-system acceptance.

RI-J/direct-K response export is explicitly rejected until a matching dual
provider response contract exists. Existing RI-JK/direct response modes are
retained. Full native auxiliary-geometry AD and noncollinear GGA remain outside
the capabilities established by this checkpoint. No merge or push is performed.
