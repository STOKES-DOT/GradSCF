"""Durable public-data transfer entry point; no electronic-structure calculation."""
import sys
from gradscf.model.autoxc.workflows.xc100.distributed import synchronize_data

if __name__ == '__main__':
    synchronize_data(sys.argv[1],sys.argv[2],sys.argv[3],status_path=sys.argv[4])
