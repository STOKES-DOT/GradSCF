"""Retry aggregation must preserve raw data and reject energy regressions."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'comparisons'))
from nnao_reactivity.merge import merge_records


def test_merge_overrides_selected_results_without_mutating_parent():
    parent = [dict(key='a', status='failed'), dict(key='b', status='passed', energy_hartree=-2.)]
    merged = merge_records(parent, [dict(key='a', status='passed', energy_hartree=-1.)])
    assert merged[0]['status'] == 'passed'
    assert merged[1] == parent[1]
    assert parent[0]['status'] == 'failed'


def test_failed_revalidation_remains_explicit_instead_of_silently_keeping_a_pass():
    parent = [dict(key='a', status='passed', energy_hartree=-2.)]
    merged = merge_records(parent, [dict(key='a', status='failed', error='stability unresolved')])
    assert merged[0]['status'] == 'failed'
    assert parent[0]['status'] == 'passed'


def test_repair_may_not_raise_an_accepted_species_energy_or_add_unknown_cases():
    parent = [dict(key='a', status='passed', energy_hartree=-2.)]
    with pytest.raises(ValueError, match='higher'):
        merge_records(parent, [dict(key='a', status='passed', energy_hartree=-1.)])
    with pytest.raises(ValueError, match='Unknown'):
        merge_records(parent, [dict(key='x', status='passed')])


def test_continuation_keeps_accepted_cases_and_fills_only_failed_or_missing_keys():
    from nnao_reactivity.join import join_records
    previous = [dict(key='a', status='passed', energy_hartree=-2.), dict(key='b', status='failed')]
    continuation = [dict(key='b', status='passed', energy_hartree=-1.), dict(key='c', status='passed')]
    result = join_records(previous, continuation, ['a', 'b', 'c'])
    assert result[0] == previous[0]
    assert all(r['status'] == 'passed' for r in result)
    assert previous[1]['status'] == 'failed'


def test_continuation_cannot_overwrite_a_pass_or_claim_missing_work_is_complete():
    from nnao_reactivity.join import join_records
    previous = [dict(key='a', status='passed')]
    with pytest.raises(ValueError, match='accepted'):
        join_records(previous, [dict(key='a', status='passed')], ['a'])
    with pytest.raises(ValueError, match='missing'):
        join_records(previous, [], ['a', 'b'])
