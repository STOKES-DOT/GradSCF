"""XC100 candidate qualification, evaluation, and campaign entry points."""
from importlib import import_module


_EXPORTS = {
    'XC100Case': 'molecule',
    'RemoteEvaluator': 'distributed',
    'design_campaign': 'distributed',
    'continue_campaign': 'continuation',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = import_module(f'.{_EXPORTS[name]}', __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
