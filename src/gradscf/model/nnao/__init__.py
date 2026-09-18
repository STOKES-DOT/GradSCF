"""Environment-conditioned contracted Gaussian bases for GradSCF.

MACE is optional; importing basis assembly does not import the neural stack.
"""
from .basis import NeuralBasis, DirectBasis, prepare_basis, prepare_direct_basis, supported_elements

__all__=['NeuralBasis','DirectBasis','prepare_direct_basis','prepare_basis','supported_elements','MACEBasisModel','build_graph','prepare_grimme_basis']


def __getattr__(name):
    if name=='prepare_grimme_basis':
        from .grimme import prepare_grimme_basis
        return prepare_grimme_basis
    if name=='MACEBasisModel':
        from .encoder import MACEBasisModel
        return MACEBasisModel
    if name=='build_graph':
        from .graph import build_graph
        return build_graph
    raise AttributeError(name)
