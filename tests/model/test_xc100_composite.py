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
