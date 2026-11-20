"""Extend the completed independent XC100 trial to 100 total rounds."""
import json
from pathlib import Path

from gradscf.model.autoxc.action_policy import ActionPolicy
from gradscf.model.autoxc.workflows.xc100.continuation import continue_campaign


output = Path('artifacts/iks-independent-100-20261118')
previous = Path('artifacts/iks-independent-20261118-r2')
allocation = json.loads((output / 'allocation.json').read_text())
result = continue_campaign(
    allocation['host'], allocation['remote_directory'], job_id=allocation['job_id'],
    previous=previous, output=output,
    system_prompt=(previous / 'prompts' / 'system.md').read_text(),
    action_policy=ActionPolicy(), rounds=100,
)
print('Completed attempts', len(result['records']), 'retained WY', result['incumbent']['reward'])
