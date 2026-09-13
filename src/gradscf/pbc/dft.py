"""Periodic Kohn-Sham facades share the periodic HF iteration machinery."""
from .scf import _SCF


class RKS(_SCF):
    default_xc='pbe'


class UKS(RKS):
    unrestricted=True


class KRKS(RKS):
    k_sampling=True


class KUKS(UKS):
    k_sampling=True
