"""A held-out evaluation must validate its frozen winner before reading data."""
from pathlib import Path
import json
import os
import subprocess
import sys


def test_tampered_winner_is_rejected_before_reference_access(tmp_path):
    root=Path(__file__).resolve().parents[1]
    winner=tmp_path/'winner.json'
    winner.write_text(json.dumps(dict(status='frozen',sha256='tampered')))
    env=os.environ.copy();env['PYTHONPATH']=str(root/'src');env['JAX_PLATFORMS']='cpu'
    completed=subprocess.run([sys.executable,str(root/'tools/pbe_agent_loop.py'),'heldout',
        '--winner',str(winner),'--references',str(tmp_path/'never_opened.json'),
        '--output',str(tmp_path/'test_output')],env=env,capture_output=True,text=True)
    assert completed.returncode != 0
    assert 'Frozen winner hash' in completed.stderr
    assert not (tmp_path/'test_output').exists()
