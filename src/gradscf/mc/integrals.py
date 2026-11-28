"""Integral preparation shared by the molecular CASCI and CASSCF kernels."""

from gradscf.ao2mo.active import transform_active, transform_active_blocks

__all__ = ["transform_active", "transform_active_blocks"]
