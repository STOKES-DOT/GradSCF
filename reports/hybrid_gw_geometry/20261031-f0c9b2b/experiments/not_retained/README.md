# Historical candidate decision (subsequently reopened)

The reference-centered zero-step metric chart was considered after a saved
SCF residual differed from the response entry residual. Two small tests were
expected RED (4.48 s), but the phenanthrene single-factor diagnostic then
showed the same residual discrepancy with identical density and no transport.
It is not an established fix of the observed problem. No production metric
change was made at that stage. This file preserves the initial investigation
and is not an active test.

Later, a strictly reconverged checkpoint isolated a second, independent
roundoff effect from normalizing the same stationary origin again. Commit
69c8ac7 therefore introduced the checked local metric chart, with separate
tests in tests/test_orbital_zero_seed_transport.py. The historical rejection
above must not be read as the final implementation status; see report.md.
