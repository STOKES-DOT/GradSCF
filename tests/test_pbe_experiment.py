"""Small array oracles for external GGA assessment and train-only fitting."""
import jax.numpy as jnp
import numpy as np


def test_stress_checks_detect_spin_bias_and_uniform_gas_change():
    from gradscf_tools.pbe_experiment import semilocal_checks
    biased = lambda theta,rho,grad:theta[0]*rho[0]
    checks, measures = semilocal_checks(biased, jnp.array([.1]))
    assert checks['finite_sampled_derivatives']
    assert not checks['spin_interchange_symmetry']
    assert not checks['uniform_density_limit_unchanged']
    assert measures['sample_count'] > 100


def test_smooth_spin_summed_gradient_correction_passes_independent_checks():
    from gradscf_tools.pbe_experiment import semilocal_checks
    def term(theta,rho,grad):
        density = rho.sum(axis=0)
        sigma = (grad.sum(axis=0)**2).sum(axis=-1)
        return theta[0]*sigma/(1+sigma+density**2)
    checks, measures = semilocal_checks(term, jnp.array([.1]))
    assert all(checks.values())
    assert measures['uniform_density_max_abs'] == 0
    assert measures['spin_swap_max_abs'] < 1e-14


def test_train_fit_does_not_use_validation_labels(tmp_path):
    from gradscf_tools.pbe_experiment import fit_and_score
    from gradscf_tools.pbe_recording import freeze_reward_config
    from gradscf_tools.pbe_loop import build_affine_cache, AffineTerm
    from test_pbe_loop_tools import _artifact
    from gradscf_tools.pbe_loop import load_teacher_reference
    path,_ = _artifact(tmp_path)
    teacher=load_teacher_reference(path)
    cache=build_affine_cache(teacher,baseline_energy_fn=lambda p,d:jnp.asarray(.4),
        terms=(AffineTerm('constant_test_oracle',lambda p,d:jnp.asarray(1.)),),stationarity_weight=0.)
    cfg=freeze_reward_config(.2, constraint_names=['finite_fixed_reference_response'])
    result=fit_and_score({'train':[cache],'validation':[cache]},initial=[0.],
        bounds=[[-1.,1.]],reward_config=cfg)
    np.testing.assert_allclose(result['parameters'],[-.2],atol=2e-6)
    assert result['metrics']['train']['n_molecules'] == 1
    assert result['metrics']['validation']['n_molecules'] == 1
    assert set(result['metrics']) == {'train','validation'}
