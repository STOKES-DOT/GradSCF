"""Capture the matrix probe's synchronized outputs in a separate validation run."""
from pathlib import Path
import runpy
import sys


def main():
    import jax
    import numpy as np
    output = Path(sys.argv[2]).with_suffix(".npz")
    original, captured = jax.block_until_ready, []
    def observe(value):
        result = original(value)
        captured[:] = [result]
        return result
    jax.block_until_ready = observe
    try:
        runpy.run_path(str(Path(__file__).with_name("matrix.py")), run_name="__main__")
    finally:
        jax.block_until_ready = original
    np.savez(output, *(np.asarray(x) for x in jax.tree.leaves(captured[0])))


if __name__ == "__main__":
    main()
