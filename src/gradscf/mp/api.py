"""Short eager MP interfaces; numerical differentiation belongs to run_mp."""
from dataclasses import fields

from ..scf.reference import (reference_from_source, unrestricted_reference_from_source,
                             is_unrestricted_source, RestrictedReference, UnrestrictedReference)
from .types import MPConfig
from .mp2 import run_mp, evaluate_mp2
from .integrals import from_scf


class RMP2:
    unrestricted = False
    method_order = 2

    def __init__(self, mf, *, frozen=None, **kwargs):
        if is_unrestricted_source(mf) != self.unrestricted:
            raise ValueError("Reference spin does not match restricted/unrestricted MP class")
        self.mf, self.frozen = mf, frozen
        cfg = MPConfig(order=self.method_order, **kwargs)
        for field in fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))
        self.result = None
        self.e_corr = self.e_tot = self.e2 = self.e3 = self.t2 = None
        self.e_corr_ss = self.e_corr_os = None
        self.converged = False

    def kernel(self):
        self.converged = False
        self.result = None
        self.e_corr = self.e_tot = self.e2 = self.e3 = self.t2 = None
        self.e_corr_ss = self.e_corr_os = None
        cfg = MPConfig(**{f.name: getattr(self, f.name) for f in fields(MPConfig)})
        if cfg.order == 3 or isinstance(self.mf, (RestrictedReference, UnrestrictedReference)):
            if not isinstance(self.mf, (RestrictedReference, UnrestrictedReference)):
                self.mf._check_reference_source()
            ref = (unrestricted_reference_from_source if self.unrestricted else reference_from_source)(self.mf)
            result = run_mp(ref.h1, ref.eri, nocc=ref.nocc,
                frozen=self.frozen, nuclear_repulsion=ref.nuclear_repulsion, config=cfg)
        else:
            result = evaluate_mp2(from_scf(self.mf, frozen=self.frozen), cfg)
        self.result = result
        self.e_corr, self.e_tot, self.e2, self.e3 = (result.correlation_energy,
            result.total_energy, result.e2, result.e3)
        self.t2 = result.t2
        self.e_corr_ss, self.e_corr_os = result.same_spin_energy, result.opposite_spin_energy
        self.converged = bool(result.valid)
        if not self.converged:
            raise RuntimeError("MP requires finite canonical HF data and resolved physical denominators")
        return self.e_corr, self.t2

    def run(self):
        self.kernel()
        return self


class UMP2(RMP2):
    unrestricted = True


class RMP3(RMP2):
    method_order = 3


def MP2(mf, **kwargs):
    return (UMP2 if is_unrestricted_source(mf) else RMP2)(mf, **kwargs)


def MP3(mf, **kwargs):
    if is_unrestricted_source(mf):
        raise NotImplementedError("MP3 currently requires a restricted canonical HF reference")
    return RMP3(mf, **kwargs)
