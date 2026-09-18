"""PySCF-style DFT namespace for GradSCF.

Lazy ``__getattr__`` dispatch (same pattern as ``gradscf/__init__.py``) so
that importing the leaf subpackage ``gradscf.dft.libxc_jax`` does not pull
in the SCF facade chain (which would create a circular import via
``scf -> model.neural_xc -> dft``).
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "RKS": "scf",
    "UKS": "scf",
    "ROKS": "dft.roks",
    "ROKSConfig": "dft.roks",
    "ROKSResult": "dft.roks",
    "run_roks_from_integrals": "dft.roks",
    "GKS": "dft.gks",
    "GKSConfig": "dft.gks",
    "GKSResult": "dft.gks",
    "run_gks_from_integrals": "dft.gks",
    "RKSConfig": "dft.rks",
    "RKSResult": "dft.rks",
    "UKSConfig": "dft.uks",
    "UKSResult": "dft.uks",
    "run_rks_from_integrals": "dft.rks",
    "run_uks_from_integrals": "dft.uks",
    "restricted_molecule_from_spec_with_jax_rks": "dft.rks",
    "CLASSIC_XC_SPECS": "dft.xc",
    "TraditionalXCFunctional": "dft.xc",
    "eval_xc_energy_density": "dft.xc",
    "eval_xc_response_tensor": "dft.xc",
    "hybrid_coeff": "dft.xc",
    "make_b3lyp_functional": "dft.xc",
    "make_classic_xc_functional": "dft.xc",
    "make_lda_functional": "dft.xc",
    "make_pbe0_functional": "dft.xc",
    "make_pbe_functional": "dft.xc",
    "parse_xc": "dft.xc",
    "semilocal_terms": "dft.xc",
    "xc_type": "dft.xc",
}

_SUBMODULES = ("xc", "rks", "uks", "roks", "gks", "libxc_jax")


def __getattr__(name: str):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    if name in _EXPORTS:
        module_name = _EXPORTS[name]
        if module_name.startswith("dft."):
            module = import_module(f".{module_name[4:]}", __name__)
        else:
            module = import_module(f"..{module_name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_EXPORTS) + list(_SUBMODULES)
