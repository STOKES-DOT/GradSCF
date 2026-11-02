# NNAO computational examples — corrected ethane supplement, 2026-10-22

Data-only handoff for the manuscript Codex. This package **supersedes Case 05
(ethane C-C stretch) in the 2026-10-21 archive**. Other cases remain unchanged.
The old archive, manuscript and paper-specific output figures were not edited.

- [Corrected case and reading order](cases/05_ethane_cc_stretch/README.md)
- [Machine-readable catalog and limitations](catalog.json)
- [Export checks](VERIFY_REPORT.json)
- [Original-source hashes](FILE_MANIFEST.json), [payload checksums](SHA256SUMS)
- [Omitted numerical/build caches](EXCLUDED_FILES.json)

The corrected set has 56 distances and nine curves: 504 accepted energy points,
112 independently checked frozen NNAO bases, 56 exact-coordinate XYZ files,
final coefficients/bases, optimization histories and corrected radial plots.
No SCF, optimization, training or finite differences were rerun for this handoff.
The existing inference-only update-3351 model is referenced in `catalog.json`
under `../2026-10-21/models/update-3351/`; it is not duplicated.

Verify this supplement without a chemistry/JAX environment:

```bash
python _tools/verify_export.py
```

Use this corrected Case 05 for new paper figures. The older
`output/ethane_cc_stretch/` still represents the historical uncorrected data;
it was deliberately left for the manuscript/figure thread to regenerate.
