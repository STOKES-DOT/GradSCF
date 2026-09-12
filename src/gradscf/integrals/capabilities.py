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

    def supports(self, operator, *, variable=None, derivative_order=0):
        if operator not in self.operators or derivative_order < 0:
            return False
        return derivative_order == 0 or (
            derivative_order <= self.max_derivative_order and variable in self.derivative_variables
        )


def backend_capabilities(name):
    operators = ("overlap", "kinetic", "nuclear", "dipole", "eri")
    if name == "native":
        return BackendCapabilities(name, operators, ("cpu",), True,
                                   representations=("cartesian", "spherical"))
    if name == "jax_reference":
        return BackendCapabilities(name, operators, ("cpu", "gpu"), True,
                                   ("centers", "nuclear_coords", "exponents", "coefficients"), 1,
                                   layouts=("full", "s4"), dtypes=("float32", "float64"),
                                   ad_modes=("jvp", "vjp"))
    if name == "gpu4pyscf":
        return BackendCapabilities(name, ("eri",), ("gpu",), False)
    raise ValueError(f"Unknown integral backend: {name!r}")
