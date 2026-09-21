# CI citation audit — 2026-09-21

Scope: method attribution in the CI README, CI kernel docstrings, the existing
`tddft/cisd.py` correction helper, and the independent singlet CIS(D) test.
This audit changes documentation only; executable ASTs were compared before
and after the edits.

| Location | Original claim | Current citation before audit | Problem type | Severity | Revision made |
| --- | --- | --- | --- | --- | --- |
| CI README, CIS(D) section | Unscaled Head-Gordon CIS(D) expression | Q-Chem 5.1 manual | Missing original-method attribution | Medium | Added HeadGordon1994; retained the manual as the inspected equation-level companion |
| `ci/solver.py`, `solve_cis` | Spin-adapted CIS equations | None | Missing method reference | Medium | Added Foresman1992 as an established CIS reference, without claiming historical priority |
| `ci/hamiltonian.py` | Slater–Condon matrix elements | Method name only | Missing foundational references | Low | Added Slater1929 and Condon1930 |
| CI hierarchy and determinant spaces | CISD/CISDT/CISDTQ and general rank truncation | Software manuals | Theory versus software attribution conflated | Medium | Added Sherrill1999 and a reference-to-code map |
| `tddft/cisd.py` docstrings | “ORCA-style” correction | No attached source or ORCA numerical comparison | Ambiguous implementation attribution | Medium | Replaced the label with a description of the spin-orbital correction and explicit scope of available validation |
| CI differentiation description | Shared implicit eigenvector response | No paper citation | Missing numerical-method context | Low | Added Xie2020 as related theory, while retaining the implementation's first-order limitation |
| PySCF comparison tests and API conventions | PySCF as reference software | API URL | Missing software-paper attribution | Low | Added Sun2020 separately from the CI/CIS(D) method citations |

## Systemic issues resolved

- Method definitions now have bibliographic entries with DOI identifiers.
- Software manuals are distinguished from original methods and reviews.
- A software comparison is not described as source-code reuse, and a literature
  citation is not presented as evidence that an untested variant is correct.
- The singlet determinant-space test is explicitly distinguished from a direct
  ORCA/Q-Chem CIS(D) calculation and from unrestricted/double-hybrid validation.

## Checks and limitations

- Seven unique BibTeX entries map to the reference registry; DOI, title and
  publication metadata were checked through publisher/author records.
- Metadata/abstract verification does not constitute a full-text,
  equation-by-equation audit of all original papers. For the CIS(D) working
  decomposition, Q-Chem 5.1 equations 7.38–7.40 were accessible and inspected.
- The older unrestricted correction and every double-hybrid use of the helper
  have not been independently validated by the RHF-singlet tests. This audit
  does not claim to close those numerical validation gaps.
- A complete historical source-code provenance/license audit is outside this
  citation check; existing project attribution is retained.
- Executable ASTs of all six edited Python files are unchanged; no numerical
  tests were rerun for these documentation-only edits.

See [REFERENCES.md](REFERENCES.md) and [references.bib](references.bib).
