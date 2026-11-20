"""Six-component XC potential search with evolutionary and independent modes."""
from importlib import import_module


_EXPORTS = {'Genome': 'genome', 'SwarmConfig': 'search', 'run_swarm': 'search'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = import_module(f'.{_EXPORTS[name]}', __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
