"""Explicit backend contracts; availability and derivative support are separate."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    operators: tuple[str, ...]
    devices: tuple[str, ...]
    jit: bool
    derivative_variables: tuple[str, ...] = ()
    max_derivative_order: int = 0
    representations: tuple[str, ...] = ("cartesian",)
    layouts: tuple[str, ...] = ("full",)
    dtypes: tuple[str, ...] = ("float64",)
    ad_modes: tuple[str, ...] = ()

    def supports(self, operator, *, variable=None, derivative_order=0, layout='full',
                 representation='cartesian',omega=0.,bounded=False):
        """Query the actual product family, not a blanket maximum order."""
        if layout not in self.layouts:return False
        if representation not in self.representations:return False
        if layout!='full' and operator!='eri':return False
        if operator not in self.operators or derivative_order < 0:
            return False
        if derivative_order==0:return True
        if operator == "ecp" and derivative_order > 0:
            return False
        if variable == "origin" and operator not in {"dipole","rinv"} and derivative_order > 0:
            return False
        if variable not in self.derivative_variables:return False
        if variable in {'coefficients','exponents'}:
            if layout!='full' or bounded:return False
            if operator=='eri' and omega:return False
            return derivative_order <= (1 if variable=='exponents' else 2)
        maximum=2
        if (representation=='cartesian' and layout=='full' and not bounded and not omega
                and operator in {'overlap','kinetic','nuclear','eri'}):
            maximum=4
        return derivative_order<=maximum


def backend_capabilities(name='libcint'):
    operators = ("overlap", "kinetic", "nuclear", "dipole", "rinv", "eri")
    if name == "libcint":
        return BackendCapabilities(name, operators+("ecp",), ("cpu",), True,
                                   ("centers", "nuclear_coords", "origin", "coefficients", "exponents"), 4,
                                   representations=("cartesian", "spherical"),
                                   layouts=("full", "s4", "s8"),
                                   ad_modes=("jvp", "vjp"))
    raise ValueError(f"Only the libcint integral interface is supported; {name!r} was removed.")
