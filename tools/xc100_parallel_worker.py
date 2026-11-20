"""Slurm entry point. Set rank affinity before importing numerical modules."""
import sys


if __name__ == '__main__':
    from gradscf.model.autoxc.workflows.xc100.parallel import run_parallel

    run_parallel(sys.argv[1])
