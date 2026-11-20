"""XC task adapter for Multi-Agent-Swarm and independent agent searches."""
import asyncio
from contextlib import ExitStack
from collections import Counter
from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from gradscf.model.autoxc import Candidate
from ..history import History, json_value, source_digest, versions, write_json
from .genome import Genome, SEMILOCAL, SLOTS
from .ppo import MutationPolicy, Transition
from .queues import AgentQueue, EvaluationQueue


ACTIONS = tuple((slot, operation) for slot in SLOTS
                for operation in ('activate', 'replace', 'remove'))
STATE_SIZE = 11


@dataclass(frozen=True)
class SwarmConfig:
    agents: int = 4
    generations: int = 5
    steps: int = 3
    ga: bool = False
    ppo: bool = False
    tournament_size: int = 3
    crossover_probability: float = .6
    seed: int = 0
    concurrency: int = 2

    def __post_init__(self):
        counts = (self.agents, self.generations, self.steps, self.tournament_size, self.concurrency)
        if any(type(x) is not int or x < 1 for x in counts):
            raise ValueError('Agent, generation, step, tournament and concurrency counts must be positive integers')
        if self.ga and self.agents < 2:
            raise ValueError('GA requires at least two agents')
        if not 0 <= self.crossover_probability <= 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError('Supply a crossover probability and a nonnegative integer seed')


def _read(path):
    return json.loads(path.read_text())


def _summary(record):
    evaluation = record.get('evaluation') or {}
    samples = tuple(evaluation.get('samples', {}).values())
    result = {k: record[k] for k in ('agent', 'generation', 'step', 'operation', 'status', 'error')}
    result['reward'] = evaluation.get('reward')
    if evaluation.get('valid') and samples:
        for key in ('density_l1', 'density_error'):
            if all(row.get(key) is not None for row in samples):
                result[key] = mean(row[key] for row in samples)
    return result


def _state(genome, evaluated, last_gain):
    summary = _summary(evaluated)
    return [1., *(float(genome.active(slot)) for slot in SLOTS),
            math.tanh(10 * summary['reward']),
            math.tanh(10 * summary.get('density_l1', 0.)),
            math.tanh(10 * summary.get('density_error', 0.)),
            math.tanh(10 * last_gain)]


def _mask(genome, registry):
    occupied = sum(genome.active(slot) for slot in SLOTS)
    return [
        (genome.active(slot) and occupied > 1 if operation == 'remove' else
         (not slot.endswith('_nonlocal') or bool(registry - SEMILOCAL))
         and (genome.active(slot) == (operation == 'replace')))
        for slot, operation in ACTIONS
    ]


def _best(trials):
    return max((r for r in trials if (r.get('evaluation') or {}).get('valid')),
               key=lambda r: r['evaluation']['reward'], default=None)


def _score(record, genome, evaluate, registry, seen):
    record['structure'] = genome.structure
    record['genome'] = genome.to_dict()
    if genome.structure in seen:
        record['status'] = 'duplicate'
        return
    try:
        candidate = genome.compile(registry)
    except ValueError as exc:
        record.update(status='invalid_proposal', error=str(exc))
        return
    seen.add(genome.structure)
    record['candidate'] = candidate.to_dict()
    record['candidate_digest'] = candidate.digest
    result, queued, occupied = evaluate(candidate)
    record.update(evaluation_queue_seconds=queued, evaluation_slot_seconds=occupied,
                  fit_seconds=result.fit.get('evaluation_seconds'))
    if result.valid and (result.reward is None or not math.isfinite(result.reward)):
        raise ValueError('A valid XC evaluation requires a finite reward')
    record['evaluation'] = json_value(result)
    record['status'] = 'evaluated'
    if result.valid:
        record['genome'] = genome.with_coefficients(result.parameters).to_dict()


def _segment(factory, evaluate, registry, config, slot, generation, parent, donor,
             previous, frozen_policy, brief):
    rng = np.random.default_rng([config.seed, generation, slot])
    last = previous[-1] if previous else {}
    session_id = last.get('session_id')
    checkpoint = frozen_policy if frozen_policy is not None else last.get('policy')
    policy = MutationPolicy(STATE_SIZE, len(ACTIONS), checkpoint) if config.ppo else None
    own_trials = [r for segment in previous for r in segment['trials']]
    seen = {r['structure'] for r in own_trials if r.get('structure')}
    seen.add(parent['structure'])
    current = parent
    trials, trajectory = [], []
    last_gain = 0.
    with ExitStack() as stack:
        agent = None
        for step in range(1, config.steps + 1):
            genome = Genome.from_dict(current['genome'])
            record = dict(agent=slot, generation=generation, step=step, operation='mutation',
                          parent=current['structure'], error=None, evaluation=None)
            trials.append(record)
            if step == 1 and donor is not None:
                donor_genome = Genome.from_dict(donor['genome'])
                differing = [s for s in SLOTS if genome.gene_structure(s) != donor_genome.gene_structure(s)]
                # Keep at least one distinct recipient gene when there is a choice.
                count = int(rng.integers(1, max(2, len(differing))))
                slots = sorted(rng.choice(differing, size=count, replace=False).tolist())
                record.update(operation='crossover', donor=donor['structure'], swapped_slots=slots)
                _score(record, genome.cross(donor_genome, slots), evaluate, registry, seen | {donor['structure']})
                if (record.get('evaluation') or {}).get('valid'):
                    current = record
                    seen.add(record['structure'])
                continue

            state, legal = _state(genome, current, last_gain), _mask(genome, registry)
            if not any(legal):
                raise ValueError('The evaluator registry leaves no valid mutation action')
            probabilities = (policy.probabilities(state, legal) if policy else
                             np.asarray(legal, dtype=float) / sum(legal))
            action = int(rng.choice(len(ACTIONS), p=probabilities))
            gene_slot, operation = ACTIONS[action]
            record['action'] = {'slot': gene_slot, 'operation': operation}
            if operation == 'remove':
                child = genome.replace_gene(gene_slot, Candidate('0'))
            else:
                feedback = {'context': brief, 'action': record['action'], 'genome': genome.to_dict(),
                            'active_inputs': sorted(registry),
                            'allowed_gene_inputs': sorted(registry & (
                                {'n'} if gene_slot.endswith('_local') else
                                SEMILOCAL if gene_slot.endswith('_semilocal') else registry)),
                            'parent_metrics': _summary(current),
                            'history': [_summary(r) for r in (own_trials + trials[:-1])[-4:]],
                            'instruction': ('Return the selected gene contribution only in the potential '
                                            'JSON schema. Supply its private definitions and parameters. '
                                            'All other genes are inherited by the controller.')}
                proposed = perf_counter()
                try:
                    if agent is None:
                        agent = stack.enter_context(factory(slot, session_id))
                    record['proposal_requested'] = True
                    record['proposal'] = json_value(agent.propose(feedback))
                    child = genome.replace_gene(gene_slot, Candidate.from_json(record['proposal']))
                    if not child.active(gene_slot):
                        raise ValueError('Activation or replacement requires a nonzero gene')
                except (ValueError, TypeError, KeyError) as exc:
                    record.update(status='invalid_proposal', error=str(exc))
                    child = None
                except (RuntimeError, OSError) as exc:
                    record.update(status='proposal_error', error=str(exc))
                    child = None
                finally:
                    record['proposal_seconds'] = perf_counter() - proposed
            if child is not None:
                _score(record, child, evaluate, registry, seen)
            if (record.get('evaluation') or {}).get('valid'):
                last_gain = record['evaluation']['reward'] - current['evaluation']['reward']
                current = record
            else:
                last_gain = -.001
            transition = Transition(state, legal, action, float(np.log(probabilities[action])), last_gain)
            trajectory.append(transition)
            record['reward'] = last_gain
        if agent is not None:
            session_id = getattr(agent, 'session_id', session_id)
    if policy and frozen_policy is None:
        policy.update(trajectory)
    return {'session_id': session_id, 'policy': policy.checkpoint() if policy else None,
            'trials': trials, 'trajectory': [asdict(t) for t in trajectory],
            'current': current}


async def run_swarm(factory, evaluator, founders, *, directory, config, registry, protocol,
                    frozen_policy=None, brief=''):
    """Evaluate a fixed number of proposals per agent in either search mode.

    Completed per-agent segments and GA generations can be replayed on restart.
    Each independent agent receives only its own trajectory and policy state.
    """
    directory, registry = Path(directory), frozenset(registry)
    founders = tuple(founders)
    if len(founders) == 1:
        founders *= config.agents
    if len(founders) != config.agents:
        raise ValueError('Supply one common founder or one founder per agent')
    if any(g.shared != founders[0].shared for g in founders):
        raise ValueError('Founders must agree on shared parameter names')
    if frozen_policy is not None and not config.ppo:
        raise ValueError('A frozen policy requires PPO mode')
    if config.ga:
        from multi_agent_ga import GAConfig, GeneticSearch, Individual

    configuration = {'config': asdict(config), 'protocol': protocol, 'registry': sorted(registry),
                     'evaluator': evaluator.configuration, 'source': source_digest(),
                     'implementation': {
                         name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('genome.py', 'ppo.py', 'queues.py', 'search.py')
                     },
                     'versions': versions(), 'founders': [g.to_dict() for g in founders],
                     'frozen_policy': frozen_policy, 'brief': brief}
    directory.mkdir(parents=True, exist_ok=True)
    agent_queue = AgentQueue(config.concurrency)
    execution = evaluator.configuration.get('worker', {}).get('execution', {})
    allocated_threads = sum(map(len, execution.get('cpu_masks', ()))) or None

    def evaluate(candidate):
        return evaluation_queue.submit(candidate).result()

    def segment_path(slot, generation):
        return directory / 'agents' / f'{slot:03d}' / f'{generation:06d}.json'

    def run_segment(slot, generation, parent, donor=None, *, agent_queue_seconds):
        path = segment_path(slot, generation)
        inputs = {'parent': parent.get('evaluation_id'), 'donor': donor and donor.get('evaluation_id')}
        if path.is_file():
            result = _read(path)
            if result['inputs'] != inputs:
                raise ValueError('Stored segment has different parents')
            return result
        previous = [_read(segment_path(slot, i)) for i in range(1, generation)]
        started = perf_counter()
        result = _segment(factory, evaluate, registry, config, slot, generation, parent, donor,
                          previous, frozen_policy, brief)
        result.update(inputs=inputs, agent_queue_seconds=agent_queue_seconds,
                      agent_slot_seconds=perf_counter() - started)
        for row in result['trials']:
            row['evaluation_id'] = f"{slot}:{generation}:{row['step']}"
        write_json(path, result)
        return result

    with (directory / '.swarm.lock').open('a') as lock, EvaluationQueue(evaluator.evaluate) as evaluation_queue:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        History(directory, configuration)
        seed_path = directory / 'founders.json'
        if not seed_path.is_file():
            seeds, assessed = [], {}
            for slot, genome in enumerate(founders):
                key = genome.compile(registry).digest
                if key not in assessed:
                    record = dict(agent=slot, step=0, generation=0, operation='founder',
                                  parent=None, error=None, evaluation_id=f'founder:{slot}')
                    _score(record, genome, evaluate, registry, set())
                    if not (record.get('evaluation') or {}).get('valid'):
                        raise ValueError('Every founder must complete the joint fit')
                    assessed[key] = record
                seeds.append({**assessed[key], 'agent': slot, 'evaluation_id': f'founder:{slot}'})
            write_json(seed_path, seeds)
        seeds = _read(seed_path)

        if config.ga:
            def individual(row):
                return Individual(row['evaluation_id'], row['structure'], row['evaluation']['reward'],
                                  True, json.dumps(row, allow_nan=False))

            async def offspring(request):
                parent = request.parent.evidence
                donor = request.donor.evidence if request.donor else None
                result = await agent_queue.run(run_segment, request.slot, request.generation, parent, donor)
                best = _best(result['trials'])
                return individual(best) if best is not None else None

            ga = GeneticSearch(GAConfig(
                # The GA core reserves the elite separately. All agent slots still get trials.
                population_size=config.agents+1, elite_count=1,
                tournament_size=min(config.tournament_size, config.agents+1),
                crossover_probability=config.crossover_probability, mutation_probability=1.,
                concurrency=config.agents, seed=config.seed), offspring,
                protocol_id=sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest())
            population = tuple(individual(seed) for seed in (*seeds, _best(seeds)))
            for generation in range(1, config.generations+1):
                # Check all committed segments immediately rather than proceeding past an I/O failure.
                if not (directory/'ga'/f'generation-{generation:08d}.json').is_file():
                    await ga.run(population, generations=generation, directory=directory/'ga')
                for slot in range(config.agents):
                    if not segment_path(slot, generation).is_file():
                        raise RuntimeError(f'Missing GA segment for generation {generation}, agent {slot}')
            await ga.run(population, generations=config.generations, directory=directory/'ga')
        else:
            async def independent(slot, parent):
                for generation in range(1, config.generations+1):
                    result = await agent_queue.run(run_segment, slot, generation, parent)
                    parent = result['current']

            outcomes = await asyncio.gather(
                *(independent(i, seed) for i, seed in enumerate(seeds)), return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome

        segments = [[_read(segment_path(slot, gen)) for slot in range(config.agents)]
                    for gen in range(1, config.generations+1)]
        trials = [r for generation in segments for segment in generation for r in segment['trials']]
        usage = []
        for slot, own_segments in enumerate(zip(*segments, strict=True)):
            measurements = [r for segment in own_segments for r in segment['trials']]
            fit_times = [r['fit_seconds'] for r in measurements if r.get('fit_seconds') is not None]
            usage.append({
                'agent': slot, 'segments': len(own_segments),
                'agent_queue_seconds': sum(s['agent_queue_seconds'] for s in own_segments),
                'evaluation_queue_seconds': sum(r.get('evaluation_queue_seconds', 0.) for r in measurements),
                'evaluation_slot_seconds': sum(r.get('evaluation_slot_seconds', 0.) for r in measurements),
                'fit_seconds': sum(fit_times) if fit_times else None,
                'measured_fits': len(fit_times),
                'allocated_cpu_seconds': allocated_threads * sum(fit_times)
                if allocated_threads is not None and fit_times else None,
            })
        result = {'mode': 'ga' if config.ga else 'independent',
                  'mutation': 'ppo' if config.ppo else 'uniform', 'agents': config.agents,
                  'scheduling': {'agent_concurrency': config.concurrency, 'evaluation_concurrency': 1,
                                 'allocated_threads': allocated_threads, 'per_agent': usage},
                  'attempts': len(trials), 'outcomes': dict(Counter(r['status'] for r in trials)),
                  'proposal_calls': sum(r.get('proposal_requested', False) for r in trials),
                  'founder_evaluations': len({r['candidate_digest'] for r in seeds}),
                  'best': _best([*seeds, *trials]), 'trials': trials}
        write_json(directory/'swarm.json', result)
        return result
