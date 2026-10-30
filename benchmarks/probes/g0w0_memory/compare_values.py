"""Compare complete captured outputs; run on the allocated numerical node."""
import json
from pathlib import Path
import sys

import numpy as np


def main():
    root, records = Path(sys.argv[1]), []
    for kind in ("qsgw", "scgw", "charge"):
        values = []
        for variant in ("baseline", "candidate"):
            name = f"{variant}-{kind}"
            measured = json.loads((root/(name+"-cpu1.jsonl")).read_text().splitlines()[0])
            validation = json.loads((root/(name+"-values.jsonl")).read_text().splitlines()[0])
            assert measured["input_sha256"] == validation["input_sha256"]
            with np.load(root/(name+"-values.npz")) as archive:
                leaves = [archive[key].copy() for key in archive.files]
            norm = sum(float(np.linalg.norm(value)) for value in leaves)
            np.testing.assert_allclose(norm, measured["norm"], atol=1e-12, rtol=1e-12)
            values.append(leaves)
        assert len(values[0]) == len(values[1])
        errors = []
        for before, after in zip(*values):
            assert before.shape == after.shape
            np.testing.assert_allclose(after, before, atol=1e-10, rtol=1e-9)
            errors.append(float(np.max(np.abs(after-before))))
        records.append(dict(kind=kind, max_abs_error_per_leaf=errors, status="ok",
                            scope="separate numerical validation, not timed samples"))
    (root/"fixed-input-numerical-check.json").write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
