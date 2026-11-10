# Full-evaluator tail counter amendment (2026-11-05)

This is an explicit solver-policy change following the remote RED of
`test_failed_fresh_switches_to_full_tail_without_duplicate_certification`
(`fresh_evaluations`: observed 2, required 1). It does not establish the cause
of the real PBE timing: the separate screen-off control gave 636.62 s / 23
cycles versus screened 635.35 s / 26, without end-to-end benefit, and its
production overlay is not retained.

Before changing assertions, the original 23-case file was preserved byte for
byte at `raw/fixtures/test_scf_reliable_refresh.initial.py`, SHA256
`ef2bde0e050b4b864d193d87db58c4fea43f7cdbf11b5b887f1e6c6dd289e568`.
The complete 24-case RED revision, including the new node, is preserved at
`raw/fixtures/test_scf_reliable_refresh.full_tail_red.py`, SHA256
`b246d6ecc7c5bb91d3979083365d6ca4e0c7a9893dd071d7b8b5dc868ad28f26`.

Only the counter/event assertions of
`test_residual_ready_rebases_drifting_energy_and_stale_best_state` change:
three main advances remain three, while extra fresh calls change from two to
one. The event sequence changes from advance/advance/fresh/advance/fresh to
advance/advance/fresh/fresh because the third main advance calls the full
evaluator with previous=None. The old fourth-event previous-state assertion
is replaced by this full-call event assertion; final and best E/F/aux checks
remain unchanged. The added RED node is unchanged. No tolerance, energy,
residual, convergence outcome, cycle bound, AD check or supported layout is
relaxed or removed.

When refresh is enabled with positive convergence tolerance, a fresh failure
replaces E/F/aux and switches the remaining main advances to evaluate(None).
The existing state_is_fresh and cycles fields encode this mode; no new carry
state, DIIS policy, stop-gradient or physical setting is introduced. Initial,
fixed-cycle and disabled-refresh behavior remain unchanged. A full main
advance is assessed with the original E/D/g gates and is not evaluated a
second time merely for certification. It counts once as an advance; only
extra checks and a necessary stale-best rebuild count as fresh_evaluations.

This amendment records implementation intent and static checks only; remote
GREEN and end-to-end scientific acceptance remain separate.

## Existing shared-solver event assertion

Remote tail-v5 validation returned 85 passed / 1 failed in 48.20 s. The only
failure was `tests/solvers/test_scf.py::test_fresh_fock_recheck_can_reject_incremental_convergence`:
its calls were `[False, False, True, True]`, while the old event assertion
expected `[False, False, True, False]`. Rejection, three main cycles and the
original gradient > 0.1 assertion all passed. This is the already declared
full-tail transition: the third main advance now calls evaluate(None).

Before amendment the complete source file was retained byte-for-byte as
`raw/fixtures/test_solvers_scf.pre_full_tail_event_amendment.py`, SHA256
`60916398d52ba0554e2de2fe0a6028133917905e19ed257bfce1fb4527a96003`.
The sole test change is the fourth event from False to True. Every numerical,
convergence, cycle-count, gradient-call-count and disabled-refresh assertion
is unchanged; production is unchanged. A complete GREEN rerun remains pending.
