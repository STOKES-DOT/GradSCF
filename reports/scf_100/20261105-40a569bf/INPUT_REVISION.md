# Input registration, 2026-11-05

No scientific calculation was executed during this input-generation step.

The initial `plan/` draft registered RI versus conventional direct tasks before
the public RI-J/direct-K keyword was settled. Its protocol hash was
`c7da9e9b275b269698367598b422d5844748ffedeabfad803cca0a43d334896b`.
It is retained as preparation provenance and must not be used as a numerical
baseline.

The authoritative `plan-v2/` adds the explicit `only_dfj` boolean and records
RI-J for pure semilocal XC, RI-JK for HF/global hybrid, and RI-J/direct-K for RSH.
Its protocol hash is
`3ce5657527478ea592e46ab4d4a13d70e8f3503a3298ce7575d077269ac820a2`.
The physical geometry is unchanged, with canonical geometry hash
`4d4f50763390ba75438c1e28e9a3774e51648af173c2eaf8cf2ab697e975008d`.

`plan-v2/input_manifest.json` contains every exact configuration-file SHA256,
the source catalog hash, source geometry identity and the complete protocol.
The matrix has 60 independent configurations (30 per engine) and at most
84 SCF calls if all capabilities and cold calls pass.

Only metadata and interface safety were checked locally: generator RED/GREEN,
4 metadata regressions, including byte-exact regeneration of all 60 configurations, standard-library CLI loading, and explicit rejection
without Slurm before numerical imports. Scientific baseline/candidate results
must be recorded separately after remote execution. The unchanged original
60-system campaign is not used as this new task's timing denominator.
