"""Slurm entry point; all numerical work lives in src."""
from pathlib import Path
import sys

from gradscf.model.autoxc.workflows.xc100.campaign import serve


if __name__ == '__main__':
    serve(Path(sys.argv[1]))
