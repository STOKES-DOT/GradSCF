"""Outer density reward stays separate from the certified inner WY fit."""
from gradscf.model.autoxc import Evaluation


def test_campaign_composes_both_density_norms_without_replacing_raw_wy(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign
    campaign=Campaign(tmp_path,objective='wy_l1_l2',maxiter=500,
        gradient_tolerance=1e-6,gradient_only=True,require_convergence=True)
    campaign.normalization=(.002,.003,.00002)
    campaign.rows=[{'system_id':'one','nelec':2}]
    sample={'variational_gap':.004,'density_l1':.003,'density_error':.00002**.5,
            'reward':.04,'valid':True}
    result=campaign.compose(Evaluation(.04,True,samples={'one':sample}))
    assert abs(result.reward)<1e-14
    assert result.samples['one']['reward']==.04
    assert result.samples['one']['composite_reward']==result.reward


def test_campaign_freezes_gradient_and_outer_objective_configuration(tmp_path):
    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign
    campaign=Campaign(tmp_path,objective='wy_l1_l2',maxiter=500,
        gradient_tolerance=1e-6,gradient_only=True,require_convergence=True)
    assert campaign.fit_options['gradient_only'] and campaign.fit_options['require_convergence']
    assert campaign.policy['objective']=='wy_l1_l2'
    assert campaign.policy['outer_weights']==[.5,.25,.25]


def test_joint_reward_preserves_wu_yang_and_ranks_energy_accuracy(tmp_path):
    import pytest

    from gradscf.model.autoxc.workflows.xc100.campaign import Campaign

    campaign = Campaign(tmp_path, candidate_mode='open_xc')
    campaign.normalization = (.002, .003, .00002)
    campaign.energy_normalization = .05
    campaign.rows = [{'system_id': 'one', 'nelec': 2, 'reference_xc_energy_hartree': -1.}]
    raw = {'delta': .002, 'reward': 1., 'valid': True}

    result = campaign.compose(Evaluation(1., True, samples={
        'one': {**raw, 'xc_energy_hartree': -1.04}}))
    worse = campaign.compose(Evaluation(1., True, samples={
        'one': {**raw, 'xc_energy_hartree': -.9}}))
    baseline = campaign.compose(Evaluation(0., True, samples={
        'one': {**raw, 'delta': 0., 'reward': 0., 'xc_energy_hartree': -.9}}))

    assert result.reward == pytest.approx(.55)
    assert worse.reward == pytest.approx(.25)
    assert baseline.reward == pytest.approx(0.)
    assert result.samples['one']['reward'] == 1.
    assert result.samples['one']['xc_energy_error_hartree'] == pytest.approx(-.04)
    assert result.fit['inner_objective'] == 'wu_yang'


def test_q_path_energy_scales_raw_density_gradient_and_laplacian():
    import jax.numpy as jnp
    import numpy as np

    from gradscf.model.autoxc import Candidate, Parameter
    from gradscf.model.autoxc.workflows.xc100.molecule import q_path_energy_density

    candidate = Candidate('-a*n**(1/3)+2*sigma+3*lap',
                          parameters=(Parameter('a', 1., 0., 2.),))
    rho = jnp.array([.2, 1.])
    gradient = jnp.array([[.4, .6, .8], [0., 0., 0.]])
    laplacian = jnp.array([-.3, .3])
    values = q_path_energy_density(candidate, {'a': 1.}, rho, gradient, laplacian, 1e-12)
    expected = rho*(-.75*rho**(1/3) + 2*jnp.sum(gradient**2, axis=-1)/3 + 1.5*laplacian)
    np.testing.assert_allclose(values, expected, atol=1e-12, rtol=1e-12)


def test_joint_worker_publishes_fixed_scales_and_joint_selection(tmp_path, monkeypatch):
    import json
    import pytest

    from gradscf.model.autoxc.workflows.xc100 import campaign as workflow

    monkeypatch.setattr(workflow, 'IDS', ('one', 'two'))
    campaign = workflow.Campaign(tmp_path, candidate_mode='open_xc')
    monkeypatch.setattr(campaign, 'validate_prepared', lambda: 0.)
    campaign.rows = [
        {'system_id': identifier, 'nelec': electrons, 'fingerprint': identifier,
         'G_hartree': gap, 'density_l1': .01, 'density_error': .01,
         'pbe_output_xc_energy_hartree': pbe, 'reference_xc_energy_hartree': reference}
        for identifier, electrons, gap, pbe, reference in
        [('one', 2, .004, -1.1, -1.), ('two', 4, .016, -2.8, -3.)]
    ]
    campaign.finish_preparation(receipt=None)

    ready = json.loads((tmp_path/'ready.json').read_text())
    assert 'selection' not in ready
    assert ready['policy']['selection'] == 'joint_reward'
    worker = ready['configuration']['worker']
    assert worker['outer_normalization'][0] == pytest.approx(.003)
    assert worker['xc_energy_normalization_hartree_per_electron'] == pytest.approx(.05)
