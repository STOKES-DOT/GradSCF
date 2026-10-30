"""Explicit dispatch between distinct offline density-inversion methods."""


def invert(target, potential, base_fock, *, method, **kwargs):
    if method == 'thermal_matrix':
        from .thermal import invert as solve
        return solve(target, potential, base_fock, method=method, **kwargs)
    if method in ('wu_yang', 'density_match'):
        from .local import invert as solve
        return solve(target, potential, base_fock, objective_kind=method, **kwargs)
    raise NotImplementedError('Use method=thermal_matrix, wu_yang or density_match explicitly')
