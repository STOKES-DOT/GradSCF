"""PySCF-style ground-state CC facades, without solver copies."""

from dataclasses import fields
from ..scf.reference import (reference_from_source, UnrestrictedReference,
                             unrestricted_reference_from_source, is_unrestricted_source)
from ..integrals.mo import frozen_indices, unrestricted_frozen_indices
from ..scf.reference import reference_state_signature
from .types import CCConfig
from .ground import run_cc
from .uccsd import run_ucc
from .lambda_equations import solve_lambda
from .triples import evaluate_triples
from .properties import _density_from_lambda, _rdm2_from_lambda


class CC:
    def __init__(self, mf, *, method="ccsd", frozen=None, **kwargs):
        cfg = CCConfig(method=method, **kwargs)
        self.mf = mf
        self.frozen = frozen
        for f in fields(cfg):
            setattr(self, f.name, getattr(cfg, f.name))
        self.result = self.reference = None
        self.e_corr = self.e_tot = self.t1 = self.t2 = None
        self.l1 = self.l2 = self.lambda_result = None
        self.converged = self.converged_lambda = False
        self._state_key = None

    def _physical_state_key(self):
        ref = self.reference
        frozen = (unrestricted_frozen_indices(ref.h1[0].shape[0], ref.nocc, self.frozen)
                  if isinstance(ref, UnrestrictedReference)
                  else frozen_indices(ref.h1.shape[0], ref.nocc, self.frozen))
        return (
            self.method,
            frozen,
            reference_state_signature(self.mf),
        )

    def _config(self):
        return CCConfig(**{f.name: getattr(self, f.name) for f in fields(CCConfig)})

    def kernel(self, t1=None, t2=None):
        unrestricted = is_unrestricted_source(self.mf)
        self.reference = ref = (unrestricted_reference_from_source(self.mf)
                                if unrestricted else reference_from_source(self.mf))
        self.result = (run_ucc if unrestricted else run_cc)(
            ref.h1,
            ref.eri,
            nocc=ref.nocc,
            nuclear_repulsion=ref.nuclear_repulsion,
            frozen=self.frozen,
            config=self._config(),
            t1=t1,
            t2=t2,
        )
        self.e_corr, self.e_tot = (
            self.result.correlation_energy,
            self.result.total_energy,
        )
        self.t1, self.t2 = self.result.t1, self.result.t2
        self.converged = bool(self.result.converged)
        self.l1 = self.l2 = self.lambda_result = None
        self.converged_lambda = False
        self._state_key = self._physical_state_key()
        return self.e_corr, self.t1, self.t2

    def run(self):
        self.kernel()
        return self

    def _ready(self):
        if self.result is None or not self.converged:
            raise RuntimeError("Converge the CC ground state first")
        if self._physical_state_key() != self._state_key:
            raise RuntimeError(
                "CC reference, method or frozen space changed; run kernel() again"
            )
        return self.reference

    def solve_lambda(self):
        ref = self._ready()
        self.lambda_result = solve_lambda(
            ref.h1,
            ref.eri,
            self.result,
            nocc=ref.nocc,
            frozen=self.frozen,
            config=self._config(),
        )
        self.l1, self.l2 = self.lambda_result.l1, self.lambda_result.l2
        self.converged_lambda = bool(self.lambda_result.converged)
        return self.l1, self.l2

    def triples(self, *, variant="ccsd(t)"):
        """Return a correction with residual, canonicality and denominator diagnostics."""
        ref = self._ready()
        if self.method != "ccsd":
            raise ValueError("Noniterative triples require CCSD amplitudes")
        result = evaluate_triples(
            ref.h1,
            ref.eri,
            self.result,
            nocc=ref.nocc,
            frozen=self.frozen,
            denominator_tol=self.denominator_tol,
            residual_tol=self.residual_tol,
            variant=variant,
        )
        if not bool(result.valid):
            raise ValueError(
                "Triples require matching converged CCSD amplitudes, canonical orbitals and resolved denominators"
            )
        return result

    def ccsd_t(self):
        return self.triples().energy

    def make_rdm1(self):
        ref = self._ready()
        self.solve_lambda()
        if not self.converged_lambda:
            raise RuntimeError("Converge Lambda before evaluating the CC density")
        return _density_from_lambda(
            ref.h1,
            ref.eri,
            self.result,
            self.lambda_result,
            nocc=ref.nocc,
            frozen=self.frozen,
            config=self._config(),
        )

    def make_rdm2(self):
        ref = self._ready()
        self.solve_lambda()
        if not self.converged_lambda:
            raise RuntimeError("Converge Lambda before evaluating the CC density")
        return _rdm2_from_lambda(ref.h1, ref.eri, self.result, self.lambda_result,
                                 nocc=ref.nocc, frozen=self.frozen, config=self._config())


class CCSD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="ccsd", **kwargs)


class RCCSD(CCSD):
    def __init__(self, mf, **kwargs):
        if is_unrestricted_source(mf):
            raise ValueError("RCCSD requires a restricted closed-shell reference")
        super().__init__(mf, **kwargs)


class UCCSD(CCSD):
    def __init__(self, mf, **kwargs):
        if not is_unrestricted_source(mf):
            raise ValueError("UCCSD requires a UHF/ROHF or UnrestrictedReference")
        super().__init__(mf, **kwargs)


class UCCD(CC):
    def __init__(self, mf, **kwargs):
        if not is_unrestricted_source(mf):
            raise ValueError("UCCD requires a UHF/ROHF or UnrestrictedReference")
        super().__init__(mf, method="ccd", **kwargs)


class CCD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="ccd", **kwargs)


class CCS(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="ccs", **kwargs)


class CC2(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="cc2", **kwargs)


class LCCD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="lccd", **kwargs)


class LCCSD(CC):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, method="lccsd", **kwargs)
