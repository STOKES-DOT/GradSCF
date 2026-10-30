"""Adaptive winner selection must never consume test metrics or another protocol."""
import copy
import pytest


def records():
    rows=[]
    for name,reward in [('c000',.85),('c001',1.1),('c002',1.05)]:
        rows.append(dict(candidate_id=name,status='evaluated',parameters=[] if name=='c000' else [.1],
            reference_protocol_sha256='protocol',reward_sha256='reward',module_sha256=None if name=='c000' else name,
            scores=dict(metrics={'train':{'valid':True},'validation':{'valid':True}},
                        reward={'penalized':reward,'reward_sha256':'reward'})))
    return rows


def test_winner_uses_frozen_validation_reward_and_includes_unmodified_baseline():
    from gradscf_tools.pbe_recording import freeze_validation_winner,canonical_hash
    values=records();frozen=freeze_validation_winner(values)
    assert frozen['candidate_id']=='c001' and frozen['status']=='frozen'
    assert frozen['protocol_sha256']=='protocol' and frozen['parameters']==[.1]
    assert frozen['sha256']==canonical_hash({k:v for k,v in frozen.items() if k!='sha256'})
    for r in values[1:]:r['scores']['reward']['penalized']=.8
    assert freeze_validation_winner(values)['candidate_id']=='c000'


def test_test_metrics_or_mixed_reward_or_missing_feedback_round_are_rejected():
    from gradscf_tools.pbe_recording import freeze_validation_winner
    values=records();values[1]['scores']['metrics']['test']={'valid':True}
    with pytest.raises(ValueError,match='test|train|validation'):freeze_validation_winner(values)
    values=records();values[1]['reward_sha256']='changed'
    with pytest.raises(ValueError,match='reward|protocol'):freeze_validation_winner(values)
    with pytest.raises(ValueError,match='two|round|baseline'):freeze_validation_winner(records()[:2])
