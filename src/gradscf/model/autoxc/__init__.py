"""Optional agent proposals around fixed, independently testable iKS evaluation."""
from importlib import import_module

_EXPORTS = {'Candidate': 'candidate', 'Parameter': 'candidate',
            'CodexDesigner': 'codex', 'Evaluator': 'loop', 'Evaluation': 'loop', 'run': 'loop'}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module('.'+_EXPORTS[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = list(_EXPORTS)
