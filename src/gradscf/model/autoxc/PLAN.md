# AutoXC infrastructure refactor

Approved scope, 2026-11-16: deterministic Python orchestration around an optional
official Codex Python client; physical operations remain in GradSCF iKS. Work
only in feat/xc-agent-design-20261116. Leave the AutoXC vendor and other branches
untouched. No live model calls, molecular jobs, merge or push in this phase.

- [x] Direct Wu–Yang assessment: one XC contribution, shared isolated-subspace
  solver, fixed-target energy-to-potential AD, gauge and complete-batch validity.
- [x] Explicit analytic candidates: safe expression parsing, serialized formulas,
  parameter bounds and existing custom_xc input/energy contracts.
- [x] Optional Codex adapter and deterministic proposal/evaluate/feedback loop:
  structured responses, immutable scoring configuration, replay without an SDK.
- [x] Durable records, restart/failure tests and a readable, SCF-free example;
  run focused CPU/float64 regressions and document unverified production paths.

Tests must reproduce the old double-XC and parameter-AD failures independently.
No candidate may gain reward by dropping failed samples. No reference potential
or inverse orbitals are required for the direct WY objective. Existing reward
and inversion APIs retain their semantics. Potential-only formulas are not
advertised as integrable energy functionals. Source/score fingerprints gate
restart; failures and all attempted formulas remain in the history.

## Local single-molecule acceptance (subsequently authorized)

The user approved a real molecule and Codex loop, then explicitly moved the
numerical work to local CPU. No remote numerical job was submitted.

- [x] Bind qualified XC100010 BH3 without altering basis, grid or target density.
- [x] Validate handwritten external PBE/custom_xc energy and potential against
  an independent libxc weak-form oracle and validate AD by finite differences.
- [x] Run two real GPT-5.6 Luna/high proposals in one official SDK thread;
  fit using AD, evaluate via iKS, and return the first reward before revision.
- [x] Preserve formulas, parameters, feedback, source manifest and score receipts;
  verify 208 tests and independent read-only review. Record single-target limits.

Measured details: examples/iks/XC100_CODEX_SINGLE.md. Scope remains local CPU,
fixed target/no candidate SCF. Broader datasets, GPU, merging and pushing were
not part of this acceptance.
