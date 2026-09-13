import numpy as np


def test_rotation_guesses_preserve_metric_and_are_seeded():
    from gradscf.scf.init_guess import orbital_rotation_guesses
    s=np.diag([.5,1.,2.,3.]);c=np.diag(np.diag(s)**-.5)
    guesses=orbital_rotation_guesses(c,amplitudes=(0.,.1),seed=71)
    assert len(guesses)==2
    np.testing.assert_allclose(guesses[0],c,atol=0,rtol=0)
    for g in guesses:np.testing.assert_allclose(g.T@s@g,np.eye(4),atol=2e-14,rtol=0)
    repeat=orbital_rotation_guesses(c,amplitudes=(0.,.1),seed=71)
    np.testing.assert_array_equal(guesses[1],repeat[1])
    assert np.linalg.norm(guesses[1]-guesses[0])>.01


def test_uks_and_uhf_facades_forward_orbital_gradient_tolerance():
    from gradscf import gto,scf
    mol=gto.M(atom='H 0 0 0',basis='sto-3g',spin=1)
    for cls in (scf.UKS,scf.UHF):
        mf=cls(mol,conv_tol_grad=2e-8)
        assert mf._config().conv_tol_grad==2e-8
