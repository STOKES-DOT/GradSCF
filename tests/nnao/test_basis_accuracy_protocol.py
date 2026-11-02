"""Physical counting and scoring gates for the 27-basis benchmark."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "comparisons/nnao_basis_accuracy/protocols.py"
spec = importlib.util.spec_from_file_location("basis_accuracy_protocol", SOURCE)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_general_contractions_share_primitives_but_not_aos():
    shells = [[[0, [1., 1., 0.], [.5, 0., 1.]], [0, [1., 1.]], [1, [.25, 1.]]]]
    counts = protocol.basis_counts(shells)
    assert counts["nao_spherical"] == 6
    assert counts["primitive_radial_unique"] == 3
    assert counts["primitive_spherical_unique"] == 5


def test_ghosts_do_not_freeze_additional_mp2_core_orbitals():
    assert protocol.frozen_core_count([8, 1, 1, 0, 0]) == 1
    assert protocol.frozen_core_count([6, 6, 1, 1]) == 2


def test_balanced_score_and_no_silent_missing_case():
    assert protocol.balanced_mae([1.] * 6, [-3.] * 6) == 2.
    with pytest.raises(ValueError):
        protocol.balanced_mae([1.] * 5, [-3.] * 6)
    with pytest.raises(ValueError):
        protocol.balanced_mae([float("nan")] * 6, [0.] * 6)


def test_model_reference_is_separate_from_conditional_cp_and_mp2_is_not_variational():
    r = protocol.cp_estimate(-10., (-4., -5.), (-4.1, -5.2), (-4.2, -5.3))
    assert r["model_interaction_hartree"] == pytest.approx(-1.)
    assert r["cp_estimate_hartree"] == pytest.approx(-.8)
    assert r["conditional_cp_hartree"] == pytest.approx(-.5)
    # A sign reversal of the MP2 BSSE estimate is permitted, not discarded.
    r = protocol.cp_estimate(-10., (-4., -5.), (-4., -5.), (-3.9, -4.9))
    assert r["bsse_estimate_hartree"] == pytest.approx(-.2)


def test_interleaved_dataset_atom_order_does_not_change_fragments():
    row = dict(symbols=['O','O','H','H','H','H'], coords_angstrom=
        [[0,0,0],[4,0,0],[0,.96,0],[.93,-.24,0],[4,.96,0],[4.93,-.24,0]])
    assert protocol.fragment_indices(row) == [[0,2,3],[1,4,5]]


def test_gpu_count_uses_allocated_tres_not_missing_squeue_gres():
    path=SOURCE.parents[3]/'reproducibility/nnao/basis_accuracy/controller.py'
    s=importlib.util.spec_from_file_location('basis_accuracy_controller',path)
    c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
    assert c.parse_gpu_count('JobState=RUNNING TRES=cpu=16,mem=32G,gres/gpu=1 Gres=(null)')==1
    assert c.parse_gpu_count('AllocTRES=cpu=16,gres/gpu=2,gres/gpu:NV5090=2')==2
    assert c.parse_gpu_count('TRES=cpu=4,mem=16G,node=1')==0
    with pytest.raises(ValueError):c.parse_gpu_count('unexpected response')
