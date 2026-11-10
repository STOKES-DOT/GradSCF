"""Explicit dispatch between distinct offline density-inversion methods."""


def invert(target, potential=None, base_fock=None, *, method, **kwargs):
    if base_fock is None:
        raise ValueError("Supply a fixed base_fock Hamiltonian")
    if method == 'zmp':
        from .zmp import invert as solve
        if potential is not None:
            kwargs.setdefault('grid', potential.grid)
        return solve(target, base_fock, **kwargs)
    if method == 'thermal_matrix':
        from .thermal import invert as solve
        return solve(target, potential, base_fock, method=method, **kwargs)
    if method in ('wu_yang', 'density_match'):
        from .local import invert as solve
        return solve(target, potential, base_fock, objective_kind=method, **kwargs)
    raise NotImplementedError('Use method=thermal_matrix, wu_yang, density_match or zmp explicitly')
