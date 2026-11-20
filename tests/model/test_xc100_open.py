"""The strict-iKS experiment exposes open inputs without importing energy labels."""
import pytest


def test_open_campaign_has_no_four_parameter_or_uniform_shape_policy(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign
    campaign=Campaign(tmp_path,candidate_mode='open',objective='wu_yang',maxiter=500,
        gradient_tolerance=1e-6,gradient_only=True,require_convergence=True)
    assert campaign.policy['max_parameters'] is None
    assert 'tau' in campaign.policy['features'] and 'virtual_0' in campaign.policy['features']
    assert campaign.policy['uniform_limit_tolerance'] is None
    assert campaign.policy['reference_energy_labels'] is False


def test_open_campaign_cannot_silently_select_a_density_weighted_reward(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign
    with pytest.raises(ValueError):Campaign(tmp_path,candidate_mode='open',objective='wy_l1_l2')


def test_open_xc_allows_full_semilocal_energy_without_frozen_orbital_fields(tmp_path):
    from gradscf.model.autoxc import Candidate
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign

    campaign = Campaign(tmp_path, candidate_mode='open_xc')
    assert campaign.policy['kind'] is None
    assert campaign.policy['reference_energy_labels'] is False
    assert campaign.bind(Candidate('pbe_x+pbe_c', kind='energy'), ()) == {}
    assert campaign.bind(Candidate('virtual_0*n', kind='potential'), ()) == {}
    with pytest.raises(ValueError, match='semilocal'):
        campaign.bind(Candidate('virtual_0*n', kind='energy'), ())


def test_open_input_provider_reuses_the_case_density_fields():
    from types import SimpleNamespace
    import jax.numpy as jnp
    import numpy as np
    from gradscf.model.autoxc.workflows.xc100.molecule import XC100Case
    from gradscf.model.autoxc.workflows.xc100.ingredients import features_for_candidate
    case=SimpleNamespace(floor=1e-12,fields=XC100Case.fields,arrays={
        'rho':jnp.array([.1,.2,.3]),'grad_rho':jnp.ones((3,3))*.02,
        'laplacian_rho':jnp.array([.01,.02,.03]),'grid_weights':jnp.ones(3)})
    case.inputs=XC100Case.inputs.__get__(case)
    fields=features_for_candidate(case,{'n','u','q'})
    assert set(fields)=={'n','u','q'}
    assert all(np.isfinite(v).all() for v in fields.values())


def test_standalone_potential_uses_density_fields_and_zero_operator_offset(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np
    from gradscf.model.autoxc import Candidate, bindings
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign
    from gradscf.model.autoxc.workflows.xc100.molecule import XC100Case

    campaign = Campaign(tmp_path, candidate_mode='standalone_potential')
    assert campaign.policy['kind'] == 'potential'
    assert campaign.policy['potential_is_complete']
    assert campaign.policy['features'] == sorted(XC100Case.fields)
    assert 'orbital_reference' not in campaign.policy
    for candidate in (Candidate('n', kind='energy'), Candidate('n*virtual_0')):
        with pytest.raises(ValueError, match='local potential and declared physical inputs'):
            campaign.bind(candidate, ())

    case = SimpleNamespace(
        system_id='010', max_parameters=None, floor=1e-12, fields=XC100Case.fields,
        arrays=dict(rho=jnp.array([.1]), grad_rho=jnp.array([[.2, .0, .0]]),
                    laplacian_rho=jnp.array([.3]), grid_weights=jnp.ones(1)),
        problem=object(), grid=object(), reporter=object(), transform=jnp.eye(2),
        vxc=jnp.eye(2)*5, timings={},
    )
    case.inputs = XC100Case.inputs.__get__(case)
    case.bind = XC100Case.bind.__get__(case)
    campaign.cases['010'] = case
    monkeypatch.setattr(campaign, 'status', lambda *args, **kwargs: None)

    def bind(problem, grid, candidate, inputs, *, baseline, **kwargs):
        np.testing.assert_array_equal(baseline, jnp.zeros((2, 2)))
        np.testing.assert_allclose(candidate.evaluate({}, inputs), [-.1+.3])
        return SimpleNamespace(cached_affine=True)

    monkeypatch.setattr(bindings, 'bind_local_potential', bind)
    assert campaign.bind(Candidate('-n+lap'))['010'].cached_affine
