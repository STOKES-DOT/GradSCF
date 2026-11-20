"""Run the six-gene local-XC search against an existing XC100 Slurm worker.

Use identical founders, numbers of agents, generations, and steps for ablations.
GA mode imports multi_agent_ga from Multi-Agent-Swarm's GA package.
Use the same --ppo-checkpoint for a GA-only ablation with PPO enabled.
The default founder is local-density exchange. Custom founder objects contain
a "genes" map keyed by the six slots and an optional "shared" parameter list.
Each gene uses the existing potential proposal schema.
Use the packaged swarm_campaign.json as campaign.json in the worker root.
The deployment must use the same GradSCF source on the controller and worker.
"""
import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path

from gradscf.model.autoxc import Candidate, CodexDesigner
from ...history import write_json
from ...swarm import Genome, SwarmConfig, run_swarm
from ...swarm.genome import SLOTS
from .campaign import LDA_EXCHANGE_POTENTIAL
from .distributed import RemoteEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='jy_gpu_2')
    parser.add_argument('--remote', required=True, help='Worker root with complete local potential support')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--agents', type=int, default=4)
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--steps', type=int, default=3, help='Trial slots per agent and generation')
    parser.add_argument('--concurrency', type=int, default=2, help='Concurrent complete agent segments')
    parser.add_argument('--ga', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ppo', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ppo-checkpoint', type=Path, help='Frozen policy or a saved agent segment JSON')
    parser.add_argument('--founders', type=Path, help='A six-gene genome JSON or a list of genomes')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    config = SwarmConfig(agents=args.agents, generations=args.generations, steps=args.steps,
                         ga=args.ga, ppo=args.ppo, concurrency=args.concurrency, seed=args.seed)
    checkpoint = None
    if args.ppo_checkpoint:
        if not config.ppo:
            parser.error('--ppo-checkpoint requires --ppo')
        data = json.loads(args.ppo_checkpoint.read_text())
        checkpoint = data['policy'] if 'policy' in data else data
        if not isinstance(checkpoint, dict):
            parser.error('--ppo-checkpoint must contain trained actor and critic weights')
    if args.founders:
        values = json.loads(args.founders.read_text())
        founders = [Genome.from_dict(value) for value in (values if isinstance(values, list) else [values])]
    else:
        genes = {slot: Candidate('0') for slot in SLOTS}
        genes['x_local'] = Candidate(LDA_EXCHANGE_POTENTIAL)
        founders = [Genome(genes)]

    prompts = Path(__file__).with_name('prompts')
    system = (prompts / 'system.md').read_text() + (
        '\n\n## Swarm mutation turns\n'
        'The controller supplies a six-gene parent and a selected action. '
        'Return only the selected gene as one potential JSON object. '
        'Use only the supplied search history and active inputs for this turn. '
        'The controller assembles and jointly refits the complete local potential.'
    )
    brief = (prompts / 'design.md').read_text()
    model, effort = 'gpt-5.6-luna', 'high'

    def factory(slot, session_id):
        cwd = args.output / 'designers' / f'{slot:03d}'
        cwd.mkdir(parents=True, exist_ok=True)
        return CodexDesigner(cwd=cwd, model=model, effort=effort, system_prompt=system,
                             kind='potential', session_id=session_id, timeout=None)

    guardian = RemoteEvaluator(args.host, args.remote, {}, receipt_directory=args.output/'numerical')
    try:
        evaluator = RemoteEvaluator.wait_ready(
            args.host, args.remote, receipt_directory=args.output/'numerical',
            timeout=None, job_id=args.job_id,
        )
        policy = evaluator.ready['policy']
        if policy.get('kind') != 'potential' or not policy.get('potential_is_complete'):
            raise ValueError('Swarm requires a complete local potential evaluator')
        if not policy.get('unbounded_waits') or evaluator.configuration.get('max_seconds') is not None:
            raise ValueError('Swarm requires unbounded_waits=true and max_seconds=null in the worker campaign.json')
        mode = policy.get('candidate_mode', 'standalone_potential')
        brief += f'\nActive XC100 mode: `{mode}`.\n'
        result = asyncio.run(run_swarm(
            factory, evaluator, founders, directory=args.output, config=config,
            registry=policy['features'], brief=brief, frozen_policy=checkpoint,
            protocol={'system_prompt_sha256': sha256(system.encode()).hexdigest(),
                      'model': model, 'effort': effort},
        ))
        print('Completed attempts', result['attempts'], 'best reward', result['best']['evaluation']['reward'])
    finally:
        try:
            guardian.release()
        except Exception as exc:
            write_json(args.output/'release_error.json', {'error': str(exc)})
            print('Resource release request failed:', exc, flush=True)


if __name__ == '__main__':
    main()
