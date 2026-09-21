# Adapted PySCF code

The following numerical contraction code is adapted from **PySCF 2.9.0**,
licensed under Apache License 2.0. The upstream copyright notices apply to these
adaptations. A copy of the license is distributed as [LICENSE.pyscf](LICENSE.pyscf).
Other original GradSCF files retain the repository's license.

| GradSCF file | Upstream source | Adaptation |
| --- | --- | --- |
| `_equations.py` | [`pyscf/cc/rccsd.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/rccsd.py), `update_amps` | JAX contractions and functional diagonal updates; return the physical residual instead of divided amplitude updates; real in-core blocks; no PySCF object/dependency |
| `_intermediates.py` | [`pyscf/cc/rintermediates.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/rintermediates.py), nine restricted intermediates | JAX array/contraction operations and direct full-MO blocks |
| `triples.py` | [`pyscf/cc/ccsd_t_slow.py`](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/cc/ccsd_t_slow.py), `kernel` and `r3` | Virtual triples streamed in a JAX loop; permutation-table contraction replaces the expanded 36 terms; no logger or PySCF runtime |

SHA-256 hashes of the installed upstream source files used in this adaptation:

```text
rccsd.py         35391234363b6ae26893891e18c468e636458ac209bfdad9606c15112457a552
rintermediates.py c7f3311b15c3482315bc5c8aaf570334dd71170c4b9a901ad6968bd97c903666
ccsd_t_slow.py   4fd3638e0176639781ecb138575a226912c99301a1923b60d20351eec36ac960
```

PySCF is a numerical comparison dependency in tests, not a dependency of these
production kernels. See [REFERENCES.md](REFERENCES.md) for scientific method
attribution, which is distinct from the source-code attribution above.

OpenMolcas was subsequently consulted for triples-method definitions and
iteration controls. Its inspected revision and source hashes are recorded in
[OPENMOLCAS.md](OPENMOLCAS.md). No OpenMolcas Fortran source was incorporated.
