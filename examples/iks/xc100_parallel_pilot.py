"""Ten fresh XC100 design rounds with a local agent and one Slurm allocation."""
from pathlib import Path
import json

from gradscf.model.autoxc.action_policy import ActionPolicy
from gradscf.model.autoxc.workflows.xc100.distributed import design_campaign


output = Path('artifacts/iks-independent-20261118-r2')
prompts = output / 'prompts'
job_id = json.loads((output / 'allocation.json').read_text())['job_id']
result = design_campaign(
    'jy_gpu_2', '/home/user_yuanjiao/private/homefile/gradscf-iks-independent-20261118-r2',
    output=output,
    system_prompt=(prompts / 'system.md').read_text(),
    brief=(prompts / 'design.md').read_text(),
    action_policy=ActionPolicy(),
    rounds=10,
    replay=False,
    job_id=job_id,
    queue_timeout=7 * 24 * 3600,
)
print('Completed attempts', len(result['records']), 'retained WY', result['incumbent']['reward'])
