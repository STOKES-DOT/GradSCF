"""Python owns fitting, evaluation, acceptance and recovery; agents only propose."""
from dataclasses import dataclass, field
import math
import time

import jax
import jax.numpy as jnp

from gradscf.solvers.nonlinear import minimize_eager
from .candidate import Candidate
from .history import History, json_value, source_digest, versions


@dataclass(frozen=True)
class Evaluation:
    reward: float | None
    valid: bool
    parameters: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    error: str | None = None


class Evaluator:
    """Fit explicit parameters and assess a fixed batch of iKS sample callbacks.

    bind(candidate) returns {sample_id: score_fn(params)}. Each score_fn returns
    WuYangScore via score or assess; the binding reuses existing physical input
    providers/custom_xc, and may assemble fixed AO directions once. No molecule,
    XC family, descriptor, hidden baseline or dataset split is hard-coded here.

    fit_ids select training samples. reward_ids can independently select an
    adaptive validation batch; a sealed test is never passed to this object.
    fingerprint MUST identify data, AO/grid conventions, baseline, reward scale,
    gates and callback implementation. No Python object-id caches are used.

    Streaming callbacks may provide loss_and_grad(vector), returning masked
    negative reward and its gradient in candidate.parameters order, to reuse
    a compiled program with dynamic physical arrays. diagnose_result(score)
    may add diagnostics to that same solve instead of diagonalizing again.
    """
    def __init__(self, bind, *, sample_ids, fingerprint, fit_ids=None, reward_ids=None,
                 maxiter=30, gradient_tolerance=1e-8, streaming=False,
                 gradient_only=False, require_convergence=False,max_seconds=None):
        self.bind, self.sample_ids, self.fingerprint = bind, tuple(sample_ids), fingerprint
        self.fit_ids = self.sample_ids if fit_ids is None else tuple(fit_ids)
        self.reward_ids = self.sample_ids if reward_ids is None else tuple(reward_ids)
        self.maxiter, self.gradient_tolerance = maxiter, gradient_tolerance
        if not isinstance(gradient_only,bool) or not isinstance(require_convergence,bool):
            raise ValueError('Gradient convergence options must be boolean')
        self.gradient_only,self.require_convergence = gradient_only,require_convergence
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=0):
            raise ValueError('max_seconds must be positive or None')
        self.max_seconds=max_seconds
        if not isinstance(streaming, bool):
            raise ValueError('streaming must be a boolean')
        self.streaming = streaming
        if (not callable(bind) or not isinstance(fingerprint, str) or not fingerprint or
                not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids) or
                any(not isinstance(s, str) or not s for s in self.sample_ids)):
            raise ValueError('Supply a callable, fixed unique sample IDs and an evaluator fingerprint')
        for subset in (self.fit_ids, self.reward_ids):
            if not subset or len(set(subset)) != len(subset) or not set(subset) <= set(self.sample_ids):
                raise ValueError('Fit/reward IDs must be nonempty subsets of the fixed batch')
        if (not isinstance(maxiter, int) or maxiter < 1 or
                not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0):
            raise ValueError('Supply positive optimizer limits')

    @property
    def configuration(self):
        return {'fingerprint': self.fingerprint, 'sample_ids': self.sample_ids,
                'fit_ids': self.fit_ids, 'reward_ids': self.reward_ids,
                'maxiter': self.maxiter, 'gradient_tolerance': self.gradient_tolerance,
                'streaming': self.streaming,'gradient_only':self.gradient_only,
                'require_convergence':self.require_convergence,'max_seconds':self.max_seconds}

    def evaluate(self, candidate):
        started=time.monotonic() if self.max_seconds is not None else None
        callbacks = self.bind(candidate)
        if set(callbacks) != set(self.sample_ids):
            raise ValueError('Bound sample IDs differ from the frozen batch')
        names = [p.name for p in candidate.parameters]
        unpack = lambda x: dict(zip(names, x))
        def objective(vector):
            scores = [callbacks[s](unpack(vector)) for s in self.fit_ids]
            good = jnp.all(jnp.stack([s.valid & jnp.isfinite(s.reward) for s in scores]))
            value = -jnp.mean(jnp.stack([s.reward for s in scores]))
            return jnp.where(good, value, jnp.nan)
        parameters = candidate.initial
        fit = {'success': True, 'iterations': 0, 'evaluations': 0,
               'gradient_converged':True,'projected_gradient_max':0.,
               'gradient_tolerance':self.gradient_tolerance}
        if names:
            try:
                value_and_grad = jax.jit(jax.value_and_grad(objective))
                if self.streaming:
                    if hasattr(callbacks, 'value_and_grad'):
                        value_and_grad = callbacks.value_and_grad
                    else:
                        sum_value_and_grad = stream_value_and_grad(callbacks, self.fit_ids, unpack)
                        def value_and_grad(vector):
                            value, gradient = sum_value_and_grad(vector)
                            return value / len(self.fit_ids), gradient / len(self.fit_ids)
                physical_objective=value_and_grad
                def budgeted_objective(vector):
                    if self.max_seconds is not None and time.monotonic()-started>self.max_seconds:
                        raise TimeoutError('Inner fit time budget exceeded')
                    return physical_objective(vector)
                result = minimize_eager(budgeted_objective,
                    jnp.asarray([parameters[n] for n in names]), maxiter=self.maxiter,
                    gradient_tolerance=self.gradient_tolerance,
                    bounds=[(p.lower, p.upper) for p in candidate.parameters],
                    gradient_only=self.gradient_only)
                parameters = {n: float(v) for n, v in zip(names, result.x)}
                fit = {'success': bool(result.success), 'iterations': int(result.nit),
                       'evaluations': int(result.nfev), 'message': str(result.message),
                       'gradient_max': float(jnp.max(jnp.abs(result.jac))),
                       'projected_gradient_max':result.projected_gradient_max,
                       'gradient_tolerance':self.gradient_tolerance,
                       'gradient_converged':bool(result.projected_gradient_max<=self.gradient_tolerance),
                       'criterion':'projected_gradient_inf' if self.gradient_only else 'optimizer_default'}
                if self.require_convergence and not fit['gradient_converged']:
                    return Evaluation(None,False,parameters,fit=fit,error='Inner fit did not reach gradient tolerance')
            except (ValueError, FloatingPointError,TimeoutError) as exc:
                return Evaluation(None, False, parameters, fit={'success': False,'gradient_converged':False}, error=str(exc))
        if self.max_seconds is not None and time.monotonic()-started>self.max_seconds:
            return Evaluation(None,False,parameters,
                fit={**fit,'success':False,'gradient_converged':False},error='Inner fit time budget exceeded')
        if hasattr(callbacks, 'sample_reports'):
            samples = callbacks.sample_reports(parameters)
        else:
            samples = sample_reports(callbacks, self.sample_ids, parameters)
        if set(samples) != set(self.sample_ids):
            raise RuntimeError('Sample reports do not cover the fixed batch')
        valid = all(row['valid'] and row['reward'] is not None and math.isfinite(row['reward'])
                    for row in samples.values())
        reward = (sum(samples[s]['reward'] for s in self.reward_ids) / len(self.reward_ids)
                  if valid else None)
        return Evaluation(reward, valid, parameters, samples, fit)


def stream_value_and_grad(callbacks, sample_ids, unpack):
    """Build a sum of independent sample loss/gradient callbacks."""
    def sample_objective(callback):
        def value(vector):
            score = callback(unpack(vector))
            return jnp.where(score.valid & jnp.isfinite(score.reward), -score.reward, jnp.nan)
        return jax.jit(jax.value_and_grad(value))
    # A prepared callback can supply the same masked negative
    # reward/gradient with stable compiled data arguments.
    functions = [callbacks[s].loss_and_grad
        if callable(getattr(callbacks[s], 'loss_and_grad', None))
        else sample_objective(callbacks[s]) for s in sample_ids]
    def value_and_grad(vector):
        value, gradient = jnp.asarray(0.), jnp.zeros_like(vector)
        for function in functions:
            part, tangent = function(vector)
            value, gradient = value+part, gradient+tangent
        return value, gradient
    return value_and_grad


def sample_reports(callbacks, sample_ids, parameters):
    """Collect the physical score and grid diagnostics from the same solve."""
    results = {}
    for name in sample_ids:
        callback = callbacks[name]
        result = callback(parameters)
        if hasattr(callback, 'diagnose_result') or hasattr(callback, 'diagnose'):
            diagnostic = (callback.diagnose_result(result) if hasattr(callback, 'diagnose_result')
                          else callback.diagnose(parameters))
            if bool(result.valid) and bool(diagnostic.valid):
                if not bool(jnp.allclose(jnp.asarray((result.value, result.delta, result.reward)),
                                        jnp.asarray((diagnostic.value, diagnostic.delta, diagnostic.reward)),
                                        atol=1e-9, rtol=1e-10)):
                    raise ValueError('Diagnostics changed the physical score')
            valid = result.valid & diagnostic.valid
            result = diagnostic._replace(valid=valid,
                reward=jnp.where(valid, diagnostic.reward, jnp.nan))
        results[name] = result
    samples = {s: json_value({k: getattr(v, k) for k in
               ('value', 'delta', 'reward', 'gap', 'variational_gap', 'density_error', 'density_l1', 'valid')})
               for s, v in results.items()}
    return samples


def _feedback(round_number, context, incumbent, records):
    # Formula/aggregate feedback only. Per-sample data and IDs stay in local records.
    previous = [{'round': r['round'], 'candidate': r['candidate'], 'status': r['status'],
                 'reward': r['evaluation']['reward'] if r['evaluation'] else None,
                 'valid': r['evaluation']['valid'] if r['evaluation'] else False,
                 'parameters': r['evaluation']['parameters'] if r['evaluation'] else None,
                 'error': r['error'] or (r['evaluation'].get('error') if r['evaluation'] else None),
                 'fit': r['evaluation']['fit'] if r['evaluation'] else None,
                 'diagnostics': _aggregate(r['evaluation'])} for r in records[-4:]]
    return {'round': round_number, 'context': context, 'incumbent': incumbent, 'history': previous}


def _aggregate(evaluation):
    if not evaluation or not evaluation['valid'] or not evaluation['samples']:
        return None
    rows = list(evaluation['samples'].values())
    result = {'count': len(rows)}
    for key in ('reward', 'density_error', 'density_l1', 'composite_reward', 'variational_gap'):
        values = [r.get(key) for r in rows]
        if all(v is not None and math.isfinite(v) for v in values):
            result[key] = {'mean': sum(values)/len(values), 'min': min(values), 'max': max(values)}
    return result


def run(designer, evaluator, *, directory, rounds=5, context='', initial_reward=0.,
        improvement_tolerance=1e-9, fail_fast=False, selection=None, action_policy=None):
    """Run up to a TOTAL round count; restart reuses saved completed proposals.

    designer.propose(feedback) returns the Candidate JSON contract. A provider
    failure consumes a recorded attempt but never becomes a numerical score.
    KeyboardInterrupt propagates with the last proposal checkpoint intact.
    Acceptance uses the evaluator's declared reward IDs. Initial reward 0
    denotes the declared baseline; a negative candidate cannot replace it.
    """
    if (not isinstance(rounds, int) or rounds < 0 or not math.isfinite(initial_reward) or
            not math.isfinite(improvement_tolerance) or improvement_tolerance < 0):
        raise ValueError('Supply a nonnegative round limit/tolerance and finite baseline reward')
    def configuration():
        config = {'schema': 1, 'source': source_digest(), 'evaluator': evaluator.configuration,
                'designer': getattr(designer, 'configuration', {}), 'context': context,
                'versions': versions(), 'float64': bool(jax.config.jax_enable_x64),
                'default_backend': jax.default_backend(),
                'fail_fast': fail_fast,
                'selection':None if selection is None else selection.configuration,
                'initial_reward': initial_reward, 'improvement_tolerance': improvement_tolerance}
        if action_policy is not None:
            config['action_policy'] = action_policy.configuration
        return config
    history = History(directory, configuration())
    pending = history.pending()
    saved_session = (pending.get('session_id') if pending else None)
    if saved_session is None:
        saved_session = next((r['session_id'] for r in reversed(history.records) if r['session_id']), None)
    if saved_session and hasattr(designer, 'resume') and getattr(designer, 'session_id', None) != saved_session:
        designer.resume(saved_session)
    incumbent = {'candidate': None, 'parameters': {}, 'reward': initial_reward, 'digest': None}
    for record in history.records:
        if record['accepted']:
            incumbent = {'candidate': record['candidate'],
                         'parameters': record['evaluation']['parameters'],
                         'reward': record['evaluation']['reward'], 'digest': record['digest']}
            if selection is not None:
                incumbent.update(selection.state(Evaluation(**record['evaluation'])))
    signatures = {r['digest'] for r in history.records if r['digest'] is not None}
    for number in range(len(history.records)+1, rounds+1):
        history.verify(configuration())
        started = time.monotonic()
        feedback = _feedback(number, context, incumbent, history.records)
        if action_policy is not None:
            decision = action_policy.choose(number, history.records)
            related = [history.records[i-1] for i in decision['related_rounds']]
            feedback['decision'] = decision
            feedback['related_history'] = _feedback(number, '', incumbent, related)['history']
            feedback['incumbent_round'] = next(
                (r['round'] for r in reversed(history.records) if r['accepted']), 0)
            incumbent_before = incumbent['reward']
        pending = history.pending()
        record = {'round': number, 'status': None, 'raw_proposal': None,
                  'candidate': None, 'digest': None, 'evaluation': None,
                  'accepted': False, 'error': None, 'session_id': None, 'parent': incumbent['digest']}
        try:
            if pending and pending['round'] == number and 'proposal' in pending:
                proposal = pending['proposal']
                record['session_id'] = pending.get('session_id')
            else:
                record['session_id'] = getattr(designer, 'session_id', None)
                history.write('pending.json', {'round': number, 'feedback': feedback,
                                               'session_id': record['session_id']})
                try:
                    proposal = designer.propose(feedback)
                except BaseException:
                    record['session_id'] = getattr(designer, 'session_id', None)
                    history.write('pending.json', {'round': number, 'feedback': feedback,
                                                   'session_id': record['session_id']})
                    raise
                record['session_id'] = getattr(designer, 'session_id', None)
                history.write('pending.json', {'round': number, 'proposal': proposal,
                                               'session_id': record['session_id']})
            record['raw_proposal'] = json_value(proposal)
        except Exception as exc:
            record.update(status='proposal_error', error=f'{type(exc).__name__}: {exc}')
        if record['status'] is None:
            try:
                candidate = Candidate.from_json(proposal)
                record.update(candidate=candidate.to_dict(), digest=candidate.digest)
            except (ValueError, TypeError) as exc:
                record.update(status='invalid_proposal', error=f'{type(exc).__name__}: {exc}')
        if record['status'] is None and candidate.digest in signatures:
            record['status'] = 'duplicate'
        if record['status'] is None:
            try:
                result = evaluator.evaluate(candidate)
                if not isinstance(result, Evaluation):
                    raise TypeError('Evaluator must return Evaluation')
                if result.valid and (result.reward is None or not math.isfinite(result.reward)):
                    raise ValueError('A valid evaluation requires a finite reward')
                record.update(status='evaluated', evaluation=json_value(result))
                accepted = (result.valid and result.reward > incumbent['reward']+improvement_tolerance
                            if selection is None else selection.better(result,incumbent))
                if accepted:
                    record['accepted'] = True
                    incumbent = {'candidate': candidate.to_dict(), 'parameters': result.parameters,
                                 'reward': result.reward, 'digest': candidate.digest}
                    if selection is not None:incumbent.update(selection.state(result))
            except Exception as exc:
                record.update(status='evaluation_error', error=f'{type(exc).__name__}: {exc}')
        record['seconds'] = time.monotonic()-started
        if action_policy is not None:
            record.update(decision=decision, policy_reward=incumbent['reward']-incumbent_before)
        history.append(record)
        if record['digest'] is not None:
            signatures.add(record['digest'])
        history.write('pending.json', None)
        if fail_fast and record['status'] in ('proposal_error', 'evaluation_error'):
            raise RuntimeError(f'Campaign stopped after {record["status"]}; history and proposal retained')
    summary = {'incumbent': incumbent, 'records': history.records,
               'source': history.configuration['source']}
    history.write('summary.json', summary)
    return summary


__all__ = ['Evaluation', 'Evaluator', 'run']
