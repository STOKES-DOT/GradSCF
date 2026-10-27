"""The historical first benzene solve succeeded before its process timed out."""
import importlib.util
from pathlib import Path

path=Path(__file__).parents[1]/'benchmarks/probes/tddft_execution/status.py'
spec=importlib.util.spec_from_file_location('td_execution_status',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_process_timeout_does_not_erase_completed_stage_or_certify_residual():
    records=[{'stage':'reference_metadata','converged':True},
             {'stage':'response_solve','repeat':0,'wall_seconds':318.49},
             {'stage':'end_to_end','repeat':0,'wall_seconds':352.62}]
    result=module.summarize({'status':'timeout'},records)
    assert result['completed_solves']==1
    assert result['scientific_pass'] is None
    assert result['timeout_scope']=='whole_repeated_process'
    assert not result['accepted_for_comparison']


def test_residual_and_process_must_both_pass_for_comparison():
    records=[{'stage':'reference_metadata','converged':True},{'stage':'end_to_end'}]
    assert not module.summarize({'status':'ok'},records)['accepted_for_comparison']
    assert not module.summarize({'status':'ok'},records,residuals=[1e-9,2e-7])['scientific_pass']
    good=module.summarize({'status':'ok'},records,residuals=[1e-10,2e-10])
    assert good['accepted_for_comparison'] and good['response_converged_per_root']==[True,True]
