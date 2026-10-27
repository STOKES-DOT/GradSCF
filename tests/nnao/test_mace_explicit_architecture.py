import jax
import numpy as np
import pytest
pytest.importorskip('mace_jax')
pytest.importorskip('flax')
from flax import nnx
from gradscf.model.nnao.encoder import MACEBasisModel
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph

def test_explicit_hidden_irreps_are_independent_of_edge_harmonics():
    model=MACEBasisModel(elements=(1,8),channels=4,hidden_irreps='4x0e + 4x1o',
        max_ell=3,correlation=2,num_interactions=2,radial_MLP=(8,8,8),num_bessel=8,
        num_polynomial_cutoff=6,avg_num_neighbors=2.,MLP_irreps='16x0e',gate='silu',
        basis_family='szp663_direct',rngs=nnx.Rngs(0))
    assert str(model.backbone.hidden_irreps)=='4x0e+4x1o'
    assert model.backbone.max_ell==3 and model.feature_width==20
    assert tuple(model.backbone.radial_MLP)==(8,8,8)
    atom=[('O',(0,0,0)),('H',(0,-.757,.587)),('H',(0,.757,.587))]
    layout=prepare_direct_basis(atom,basis_family='szp663_direct')
    graph=build_graph([8,1,1],[p for _,p in atom],element_order=(1,8))
    expected=np.array(layout.reference_outputs());expected/=np.maximum(np.linalg.norm(expected,axis=-1,keepdims=True),1e-30)
    np.testing.assert_allclose(model(graph),expected,atol=1e-13)
