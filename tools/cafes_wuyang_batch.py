#!/usr/bin/env python3
"""Prepare the frozen30-molecule iKS reference batch in a Slurm CPU job."""
import argparse
from pathlib import Path

from gradscf_tools.iks_workflow import run_teacher_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--seeds', help='Earlier same-input potentials as initial guesses only')
    args = parser.parse_args()
    result = run_teacher_batch(args.inputs, args.output, args.protocol,
                                Path(__file__).resolve().parents[1], seed_root=args.seeds)
    return 0 if result['status'] == 'qualified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
