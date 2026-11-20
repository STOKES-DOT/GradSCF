"""SSH/file mailbox adapter; model calls remain local, numerics stay in Slurm."""
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time

from gradscf.model.autoxc import Evaluation
from gradscf.model.autoxc.history import source_digest, write_json


SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']


class ReplayProposals:
    """Re-evaluate saved formulas after a source repair without new model calls."""
    def __init__(self,designer,seed):
        self.designer,self.seed=designer,seed
        self.proposals=tuple(seed['proposals'])

    @property
    def session_id(self):return self.designer.session_id

    @property
    def configuration(self):
        digest=hashlib.sha256(json.dumps(self.seed,sort_keys=True).encode()).hexdigest()
        return {**self.designer.configuration,'replayed_proposals':len(self.proposals),
                'replay_sha256':digest,'adapter_source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def resume(self,session):self.designer.resume(session)

    def propose(self,feedback):
        number=feedback['round']
        return self.proposals[number-1] if number<=len(self.proposals) else self.designer.propose(feedback)


class RemoteEvaluator:
    def __init__(self, host, root, ready, *, receipt_directory, timeout=3600):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', host) or not str(root).startswith('/'):
            raise ValueError('Use an SSH alias and absolute project path')
        self.host, self.root, self.ready = host, PurePosixPath(root), ready
        self.timeout = timeout
        self.receipts = Path(receipt_directory)
        self.receipts.mkdir(parents=True, exist_ok=True)
        if ready:
            self._accept_ready(ready)

    def _accept_ready(self, ready):
        if ready['baseline_count'] != 100:
            raise RuntimeError('The full dataset has not qualified')
        if ready['source'] != source_digest():
            raise RuntimeError('Local controller and numerical worker source differ')
        self.ready = ready

    @property
    def configuration(self):
        return self.ready['configuration']

    def _transport(self, command):
        """Retry connection loss; never retry a failed remote numerical command."""
        for attempt in range(8):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                error = result.stderr or ''
                permanent = any(message in error.lower() for message in
                    ('permission denied', 'host key verification failed',
                     'remote host identification has changed'))
                if result.returncode != 255 or permanent:
                    return result
            except subprocess.TimeoutExpired as exc:
                error = str(exc)
            if attempt == 7:
                raise RuntimeError(f'Worker transport failed after 8 attempts: {error[-1200:]}')
            delay = min(2**(attempt+1), 30)
            print(f'Worker connection interrupted; retry {attempt+2}/8 in {delay}s: '
                  f'{error[-180:]}', flush=True)
            time.sleep(delay)

    def _ssh(self, command, *, missing=False):
        result = self._transport(['ssh', *SSH_OPTIONS, self.host, command])
        if missing and result.returncode == 1:
            return None
        if result.returncode:
            raise RuntimeError(f'Worker SSH failed: {result.stderr[-1200:]}')
        return result.stdout

    def read(self, relative):
        path = shlex.quote(str(self.root/relative))
        text = self._ssh(f'test -f {path} && cat {path}', missing=True)
        return json.loads(text) if text else None

    def _running(self, job_id):
        if not str(job_id).isdigit():
            raise RuntimeError('Worker has no valid Slurm allocation identity')
        state = self._ssh('squeue -h -j '+str(job_id)+' -o %T').strip()
        if state != 'RUNNING':
            raise RuntimeError('Numerical worker stopped or lost its Slurm allocation')

    @classmethod
    def wait_ready(cls, host, root, *, receipt_directory, timeout=3600, job_id=None):
        client = cls(host, root, {}, receipt_directory=receipt_directory, timeout=timeout)
        if job_id is not None and not str(job_id).isdigit():
            raise ValueError('Use a numeric Slurm job ID')
        queue_command = None if job_id is None else f'squeue -h -j {job_id} -o %T'
        start, last, next_queue_check = time.monotonic(), None, 0.
        while timeout is None or time.monotonic()-start < timeout:
            now = time.monotonic()
            if queue_command is not None and now >= next_queue_check:
                queued = client._ssh(queue_command, missing=True)
                if not queued or not queued.strip():
                    raise RuntimeError(f'Slurm job {job_id} ended before publishing ready')
                next_queue_check = now + 60
            status = client.read('status.json')
            if status:
                label = (status['phase'], status.get('completed_baselines'))
                if label != last:
                    print('Numerical worker:', label, flush=True); last = label
                if status['phase'] in ('failed', 'released', 'idle_timeout'):
                    raise RuntimeError('Numerical worker stopped: '+status.get('error', status['phase']))
            ready = client.read('ready.json')
            if (ready and status and status['phase']=='ready' and
                    ready.get('job_id')==status.get('job_id')):
                if job_id is not None and str(ready['job_id']) != str(job_id):
                    raise RuntimeError(f'Unexpected Slurm worker for job {job_id}')
                client._running(ready['job_id'])
                client._accept_ready(ready)
                if timeout is not None:
                    client.timeout = 3600  # Preparation and evaluation have separate deadlines.
                (client.receipts/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
                return client
            time.sleep(60 if queue_command is not None and not status else 10)
        raise TimeoutError('Full dataset preparation did not finish in time')

    def evaluate(self, candidate):
        digest = candidate.digest
        request = {'candidate': candidate.to_dict(), 'fingerprint': self.configuration['fingerprint']}
        (self.receipts/f'{digest}.request.json').write_text(json.dumps(request, indent=2)+'\n')
        # Publish by rename so the worker never reads a partial scp upload.
        with tempfile.TemporaryDirectory(prefix='gradscf-candidate-') as folder:
            path = Path(folder)/'request.json'
            path.write_text(json.dumps(request))
            temporary = str(self.root/'inbox'/f'{digest}.tmp')
            uploaded = self._transport(['scp', *SSH_OPTIONS, str(path), f'{self.host}:{temporary}'])
            if uploaded.returncode:
                raise RuntimeError(f'Worker upload failed: {uploaded.stderr[-1200:]}')
        final = str(self.root/'inbox'/f'{digest}.json')
        # A connection can disappear after a successful rename. Replaying the
        # publication must accept the already complete content-addressed file.
        self._ssh('test -f '+shlex.quote(final)+' || mv '+shlex.quote(temporary)+' '+shlex.quote(final))
        start = time.monotonic()
        next_queue_check = start + 60
        while self.timeout is None or time.monotonic()-start < self.timeout:
            response = self.read(f'outbox/{digest}.json')
            if response:
                if response['digest'] != digest or response['fingerprint'] != self.configuration['fingerprint']:
                    raise RuntimeError('Worker response identity/fingerprint changed')
                (self.receipts/f'{digest}.response.json').write_text(json.dumps(response, indent=2)+'\n')
                result = Evaluation(**response['evaluation'])
                print('Candidate', digest[:12], 'reward', result.reward, 'valid', result.valid, flush=True)
                return result
            status = self.read('status.json')
            if status and status['phase'] in ('failed', 'released', 'idle_timeout'):
                raise RuntimeError('Numerical worker stopped: '+status.get('error', status['phase']))
            if self.timeout is None and self.ready.get('job_id') is not None:
                now = time.monotonic()
                if now >= next_queue_check:
                    self._running(self.ready['job_id'])
                    next_queue_check = now + 60
            time.sleep(5)
        raise TimeoutError('Numerical candidate evaluation timed out; saved request retained')

    def release(self):
        self._ssh('touch '+shlex.quote(str(self.root/'release.flag')))


def design_campaign(host, root, *, output, brief, rounds=50, system_prompt=None,
                    action_policy=None, replay=True, job_id=None, queue_timeout=172800,
                    candidate_mode=None):
    """Own the numerical allocation's release across every controller exit."""
    from gradscf.model import autoxc
    output = Path(output)
    proposal_directory = output/'designer'
    proposal_directory.mkdir(parents=True, exist_ok=True)
    guardian = RemoteEvaluator(host, root, {}, receipt_directory=output/'numerical')
    seed_path=output/'seed_proposals.json'
    try:
        seed=json.loads(seed_path.read_text()) if replay and seed_path.exists() else None
        # Pending Slurm allocations may outlast numerical preparation. No model
        # calls are made while queued; terminal worker states still fail fast.
        wait_options = {} if job_id is None else {'job_id': job_id}
        evaluator = RemoteEvaluator.wait_ready(host, root, receipt_directory=output/'numerical',
                                               timeout=queue_timeout, **wait_options)
        print('All 100 baselines qualified; starting real design rounds.', flush=True)
        policy = getattr(evaluator,'ready',{}).get('policy',{})
        if candidate_mode is not None and policy.get('candidate_mode') != candidate_mode:
            raise ValueError(f'This design requires numerical candidate_mode={candidate_mode}')
        kind = policy.get('kind')
        selection_spec=getattr(evaluator,'ready',{}).get('selection')
        selection=autoxc.WYSelection(**selection_spec) if selection_spec else None
        designer_options = {} if system_prompt is None else {'system_prompt': system_prompt}
        with autoxc.CodexDesigner(cwd=proposal_directory, model='gpt-5.6-luna', effort='high',
                                 timeout=300., kind=kind,session_id=seed.get('session_id') if seed else None,
                                 **designer_options) as designer:
            if seed:designer=ReplayProposals(designer,seed)
            result = autoxc.run(designer, evaluator, directory=output/'run', rounds=rounds,
                               context=brief, fail_fast=True,selection=selection,
                               action_policy=action_policy)
        (output/'final.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        try:
            guardian.release()
        except Exception as exc:
            (output/'release_error.json').write_text(json.dumps({'error':str(exc)})+'\n')
            print('Resource release request failed; inspect Slurm allocation:',exc,flush=True)


def synchronize_data(source, host, root, *, status_path, timeout=21600):
    """Resumable public archive transfer; incomplete files never become inputs."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', host) or not str(root).startswith('/'):
        raise ValueError('Use an SSH alias and absolute campaign root')
    source = Path(source)
    if not (source/'canonical_basis.json').is_file():
        raise ValueError('The qualified local dataset is unavailable')
    start = time.monotonic()
    command = ['rsync', '-a', '--partial-dir=.rsync-partial', '--progress', '--timeout=120',
        '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=6',
        '--exclude=._*', str(source)+'/', f'{host}:{root}/data/xc100/']
    for attempt in range(1, 11):
        remaining = timeout-(time.monotonic()-start)
        if remaining <= 0: break
        write_json(status_path, {'phase':'transferring', 'attempt':attempt, 'pid':os.getpid()})
        result = subprocess.run(command, timeout=remaining)
        if result.returncode == 0:
            write_json(status_path, {'phase':'complete', 'attempts':attempt,
                                    'seconds':time.monotonic()-start})
            return
        print('Public data transfer interrupted:',result.returncode,'; resuming.',flush=True)
        time.sleep(10)
    write_json(status_path, {'phase':'failed', 'seconds':time.monotonic()-start})
    raise RuntimeError('Qualified dataset synchronization did not complete')
