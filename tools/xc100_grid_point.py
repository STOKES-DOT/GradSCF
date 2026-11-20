"""Thin single-process entry point for the existing XC grid benchmark."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--block-size', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    from gradscf.model.autoxc.workflows.xc100.grid_benchmark import run_point
    result = run_point(args.root, args.id, args.block_size, repeats=args.repeats)
    print('RESULT', result['status'], result['system_id'], result['block_size'],
          result['elapsed_seconds'], flush=True)
    if result['status'] != 'ok':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
