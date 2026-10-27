"""Skipping a finite difference must be explicit and cannot erase a failed check."""
import json
import numpy as np
import pytest

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import pool_channel_keys
from gradscf.model.nnao.workflows.basis_grid import select_candidate
from gradscf.model.nnao.workflows.pool_calibration import load_references


def pilot(path):
    refs=dict(load_references(ids=['H2']),basis_family='nnao_dzp_m',core_primitives='6-31g')
    layout=prepare_direct_basis('H 0 0 0; H 0 0 .74',basis_family=refs['basis_family'],core_primitives='6-31g')
    keys=pool_channel_keys([layout])
    def write(name,value):
        (path/name).write_text(json.dumps(value))
    write('metadata.json',dict(references=refs,configuration=dict(check_gradients=False)))
    write('summary.json',dict(status='complete',comparisons=[dict(candidate='baseline',
        structure_id='H2',free_energy_hartree=-1.,pool_energy_hartree=-1.1,
        delta_contraction_hartree=.1,pool_scf=dict(converged=True,stable=True),
        free_scf=dict(converged=True,stable=True),optimizer_status='max_steps')]))
    write('baseline-parameters.json',dict(keys=keys,values=np.zeros((len(keys),2)).tolist()))
    return refs


def test_selection_records_explicitly_skipped_finite_difference(tmp_path):
    selected=select_candidate(tmp_path,pilot(tmp_path),require_gradient_check=False)
    assert selected['selection']['exponent_direction_check']=='skipped_by_request'


def test_default_selection_still_requires_a_passing_gradient_gate(tmp_path):
    with pytest.raises(ValueError,match='gradient'):
        select_candidate(tmp_path,pilot(tmp_path))


def test_failed_gradient_gate_cannot_be_relabelled_as_skipped(tmp_path):
    refs=pilot(tmp_path)
    (tmp_path/'exponent-gradient-check.json').write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError,match='gradient'):
        select_candidate(tmp_path,refs,require_gradient_check=False)
