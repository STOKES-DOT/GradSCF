"""Numerical dependency changes must invalidate a resumable experiment."""
import pytest


@pytest.mark.parametrize('dependency', ['solvers/linear/implicit.py',
                                      'solvers/nonlinear/root.py', 'tools/features.py'])
def test_source_fingerprint_covers_indirect_numerical_dependencies(tmp_path, monkeypatch, dependency):
    from gradscf.model.autoxc import history
    root = tmp_path/'gradscf'
    paths = ['model/autoxc/history.py', 'dft/iks/wu_yang.py', 'dft/iks/reward.py',
             'dft/functional.py', 'solvers/eigen/response.py',
             'solvers/nonlinear/eager.py', dependency]
    for name in paths:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# original\n')
    monkeypatch.setattr(history, '__file__', str(root/'model/autoxc/history.py'))
    original = history.source_digest()
    (root/dependency).write_text('# changed numerical implementation\n')
    assert history.source_digest() != original
