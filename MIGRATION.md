# Migrating from GradTDDFT to GradSCF

GradSCF is the current project and package name. This migration changes the
package identity and imports; it preserves the existing numerical methods,
function signatures, and scientific submodule names.

## Name mapping

| Previous name | Current name |
| --- | --- |
| Distribution `td-graddft` | Distribution `gradscf` |
| `import td_graddft` | `import gradscf` |
| `from td_graddft import gto, scf, dft` | `from gradscf import gto, scf, dft` |
| `td_graddft_tools` | `gradscf_tools` |
| `src/td_graddft/` | `src/gradscf/` |
| `src/td_graddft_tools/` | `src/gradscf_tools/` |

Replace the package prefix in notebook imports, Python scripts, `python -m`
commands, dynamic import strings, and test monkeypatch targets. For example,
`td_graddft.scf.GHF` becomes `gradscf.scf.GHF`.

No `td_graddft` or `td_graddft_tools` compatibility namespace is shipped.
An old installation may still provide those names independently; importing it
does not select this checkout.

## Install or run the current checkout

With the required dependencies already available:

```sh
python -m pip install --no-deps -e .
python -c 'import gradscf; print(gradscf.__file__)'
```

For source-based execution without changing the environment:

```sh
PYTHONPATH=src python -c 'from gradscf import gto, scf, dft; print(scf.__file__)'
python -m pytest -q tests/test_gradscf_package.py
```

On c20, the working directory remains `/home/yjiao/GradSCF`, with the existing
`/home/yjiao/opt/miniconda3/envs/jax_scf/bin/python` interpreter. The package
layout is now `src/gradscf`; the environment name does not need to change.

## Scientific names and existing data

Keep `tdscf`, `tddft`, TDA/TDDFT result names, and upstream GradDFT model or
adapter names: they identify methods or external projects. SCF class names
such as `UHF`, `ROHF`, `GHF`, `RKS`, `UKS`, `ROKS`, and `GKS` are unchanged.

The historical `reproducibility/v1.0.0/` files, their hashes, and third-party
basis data remain unchanged. NPZ target bundles retain the original embedded
metadata key so existing bundles remain readable. This does not provide
compatibility for arbitrary Python pickles containing old module paths.

The Git remote and repository URL still identify the existing GradTDDFT
repository; this API migration does not rename the GitHub repository.
