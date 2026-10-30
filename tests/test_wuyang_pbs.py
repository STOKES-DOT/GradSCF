import numpy as np

from gradscf_tools.wuyang_pbs import atom_centered_gaussian_pbs


def test_atom_centered_gaussian_pbs_has_declared_order_and_no_hidden_scaling():
    basis, settings = atom_centered_gaussian_pbs(
        np.array([[0., 0., 0.], [1., 0., 0.]]),
        exponents=(1.0,), monomials=("1", "dx", "dy", "dz"),
    )
    values = np.asarray(basis(None, np.array([[0., 0., 0.]])))
    assert settings["nparam"] == 8
    assert values.shape == (1, 8)
    np.testing.assert_allclose(values[0, :4], [1., 0., 0., 0.])
    np.testing.assert_allclose(values[0, 4:], np.exp(-1.) * np.array([1., -1., 0., 0.]))
