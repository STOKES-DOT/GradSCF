"""Dataset-specific AutoXC workflows, loaded on demand."""
from importlib import import_module


__all__ = ['xc100']


def __getattr__(name):
    if name in __all__:
        return import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
