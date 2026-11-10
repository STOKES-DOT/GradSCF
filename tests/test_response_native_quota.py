"""A response budget may be smaller than its reference SCF allowance."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from gradscf import integrals
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.tddft.response import _response_grid_view
from gradscf.tddft.response_options import plan_response_memory


def test_response_rebinds_large_scf_private_quota_without_rejecting_valid_budget():
    atom = ';'.join(f'H 0 0 {i}' for i in range(384))
    top, params = integrals.prepare_basis(atom, [[0, [.7, 1.]]], cart=True)
    original = NativeDirectBasis(integrals.make_plan(top), params)
    options = dict(direct_workers=8, integral_cache='none')
    assert plan_response_memory(384, 8, 1, np.float64, 320.,
                                direct_basis=original, **options).batch_size > 0
    large = replace(original, private_output_bytes=1024**3)
    selected = plan_response_memory(384, 8, 1, np.float64, 320.,
                                    direct_basis=large, **options)
    assert selected.batch_size > 0
    assert 0 <= selected.native_private_output_bytes < 64*1024**2
    assert large.private_output_bytes == 1024**3


def test_response_view_binds_only_execution_metadata_and_preserves_reference():
    top, params = integrals.prepare_basis('H 0 0 0; H 0 0 .8', 'sto-3g', cart=True)
    provider = NativeDirectBasis(integrals.make_plan(top), params, private_output_bytes=1024**3)
    source = SimpleNamespace(direct_basis=provider, grid_ao=None)
    view = _response_grid_view(source, 0, native_private_output_bytes=4096)
    assert view is not source and source.direct_basis is provider
    assert view.direct_basis.plan is provider.plan and view.direct_basis.parameters is params
    assert view.direct_basis.private_output_bytes == 4096
