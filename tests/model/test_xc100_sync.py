"""Interrupted public transfers resume without exposing partial archive inputs."""
import json
from types import SimpleNamespace


def test_transfer_retries_and_keeps_partial_files_separate(tmp_path,monkeypatch):
    from gradscf.model.autoxc.workflows.xc100.distributed import synchronize_data
    data=tmp_path/'qualified';data.mkdir();(data/'canonical_basis.json').write_text('{}')
    calls=[];codes=iter([20,0])
    monkeypatch.setattr('subprocess.run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(returncode=next(codes))))
    monkeypatch.setattr('time.sleep',lambda _:None)
    status=tmp_path/'transfer.json'
    synchronize_data(data,'host','/project',status_path=status)
    assert len(calls)==2 and '--partial-dir=.rsync-partial' in calls[0]
    assert json.loads(status.read_text())['phase']=='complete'
