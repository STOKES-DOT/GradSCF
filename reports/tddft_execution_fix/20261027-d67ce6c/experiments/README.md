# Ordered production patches

The five patches apply in alphabetical order to d67ce6c. Each corresponds to
the production commit in `../implementation_commits.json`; later patches can
extend a file touched earlier. They preserve the parent physics and output.

- A: stable dynamic response program identity, metadata and validation.
- B: budgeted dense/streamed occupied/virtual grid fields.
- C: public packed J/K, joint workspace and ERI transpose budgeting.
- D: explicit CPU auxiliary RI preparation and device residency.
- E: opt-in scalar solver accounting (absent from clean execution).

The source-only diagnostic helpers and report harness are committed separately.
The dense-only B prototype and early A prototype are retained in raw evidence
as superseded/intermediate runs. They are not alternate production engines.
GPU default tiling retains a measured time/memory tradeoff; no unverified
universal GPU acceleration claim or automatic tuning layer is added.

The final sources were tested remotely and hashed against the local source
files (`../source_audit.json`). Intermediate commits were organized from that
tested final tree; no separate full suite is claimed for every intermediate
commit. The source hashes and final scientific gates are authoritative.
