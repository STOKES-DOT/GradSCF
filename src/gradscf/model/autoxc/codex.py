"""A single optional official Codex SDK thread; no custom RPC or credentials."""
import json
import math
from pathlib import Path
from threading import Event, Timer

from .candidate import PROPOSAL_SCHEMA


class CodexDesigner:
    """Context-managed proposal thread. Construction/import does not start Codex.

    The SDK owns authentication and runtime transport. The designer receives
    formula/aggregate feedback, not sample matrices or held-out metrics. Its
    filesystem is read-only; deployments needing sealed-data read isolation
    must also isolate the SDK process from that dataset. No stronger boundary
    is implied by a prompt or read-only write restrictions.
    """
    def __init__(self, *, cwd, model=None, effort=None, session_id=None, timeout=300.):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Codex turn timeout must be positive and finite')
        self.cwd = str(Path(cwd).resolve())
        self.model, self.effort, self.session_id = model, effort, session_id
        self.timeout, self._active = timeout, False
        self._client = self._thread = None

    @property
    def configuration(self):
        return {'backend': 'openai-codex', 'model': self.model, 'effort': self.effort,
                'timeout': self.timeout}

    def __enter__(self):
        if self._active:
            raise RuntimeError('CodexDesigner is already open')
        self._connect()
        self._active = True
        return self

    def _connect(self):
        try:
            from openai_codex import Codex, CodexConfig, Sandbox
        except ImportError as exc:
            raise ImportError('CodexDesigner requires the optional openai-codex Python SDK') from exc
        self._client = Codex(CodexConfig(cwd=self.cwd))
        options = {'cwd': self.cwd, 'model': self.model, 'sandbox': Sandbox.read_only}
        try:
            self._thread = (self._client.thread_resume(self.session_id, **options)
                            if self.session_id else self._client.thread_start(**options))
            self.session_id = self._thread.id
        except BaseException:
            self.close()
            raise

    def resume(self, session_id):
        """Restore the persisted design thread before proposing the next formula."""
        if not self._active:
            raise RuntimeError('Use CodexDesigner as a context manager')
        self.session_id = session_id
        if self._client is None:
            self._connect()
        else:
            from openai_codex import Sandbox
            self._thread = self._client.thread_resume(session_id, cwd=self.cwd,
                model=self.model, sandbox=Sandbox.read_only)

    def propose(self, feedback):
        if not self._active:
            raise RuntimeError('Use CodexDesigner as a context manager')
        if self._thread is None:
            self._connect()
        instructions = ('Design one explicit analytic XC candidate from the feedback. '
                        'Return only the declared JSON object. Do not evaluate or edit files, '
                        'read datasets, alter scoring rules, or claim unmeasured results. '
                        'State whether the formula is an energy density or a local potential. '
                        'The Python evaluator owns parameter fitting and reward.\n')
        options = {'output_schema': PROPOSAL_SCHEMA}
        if self.effort is not None:
            options['effort'] = self.effort
        expired = Event()
        client = self._client
        def cancel():
            expired.set()
            client.close()  # Official SDK terminates its runtime and releases pending reads.
        timer = Timer(self.timeout, cancel)
        timer.daemon = True
        timer.start()
        try:
            result = self._thread.run(instructions+json.dumps(feedback, allow_nan=False), **options)
        except Exception as exc:
            if expired.is_set():
                raise TimeoutError(f'Codex turn timed out; saved thread {self.session_id}') from exc
            raise
        finally:
            timer.cancel()
            timer.join()
            if expired.is_set():
                self._client = self._thread = None
        if expired.is_set():
            raise TimeoutError(f'Codex turn timed out; saved thread {self.session_id}')
        return result.final_response  # Preserve malformed replies for the controller's audit.

    def close(self):
        if self._client is not None:
            self._client.close()
        self._client = self._thread = None

    def __exit__(self, *args):
        self.close()
        self._active = False


__all__ = ['CodexDesigner']
