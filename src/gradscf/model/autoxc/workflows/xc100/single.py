"""Design standalone local XC potentials with XC100 inverse-KS feedback."""
import argparse
from pathlib import Path

from ...action_policy import ActionPolicy
from .distributed import design_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='jy_gpu_2')
    parser.add_argument('--remote', required=True, help='Worker root for the selected candidate mode')
    parser.add_argument('--candidate-mode', choices=('standalone_potential', 'open_xc'),
                        default='standalone_potential')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--rounds', type=int, required=True)
    args = parser.parse_args()
    prompts = Path(__file__).with_name('prompts')
    result = design_campaign(
        args.host, args.remote, output=args.output, job_id=args.job_id,
        system_prompt=(prompts / 'system.md').read_text(),
        brief=(prompts / 'design.md').read_text() + f'\nActive XC100 mode: `{args.candidate_mode}`.\n',
        rounds=args.rounds, action_policy=ActionPolicy(), replay=False,
        candidate_mode=args.candidate_mode, queue_timeout=7 * 24 * 3600,
    )
    print('Completed attempts', len(result['records']), 'retained reward', result['incumbent']['reward'])


if __name__ == '__main__':
    main()
