"""Compare full saved Fock arrays and three warm samples, under declared gates."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
for name in ("before", "after", "gates", "output"):
    parser.add_argument(name, type=Path)
parser.add_argument('--native-patch', type=Path,
                    help='Explicit recorded native single-factor diff; otherwise require identical libraries.')
args = parser.parse_args()
before, after, gates = [json.loads(path.read_text()) for path in (args.before, args.after, args.gates)]
for key in ("affinity", "checkpoint_sha256"):
    assert before[key] == after[key], key
if args.native_patch is None:
    assert before['native_sha256'] == after['native_sha256']
else:
    assert args.native_patch.is_file() and args.native_patch.stat().st_size > 0
with np.load(args.before.with_suffix(".fock.npz")) as a, np.load(args.after.with_suffix(".fock.npz")) as b:
    density_error = float(np.max(np.abs(a["density"]-b["density"])))
    fock_error = float(np.max(np.abs(a["fock"]-b["fock"])))
times = [[row["seconds"] for row in value["samples"] if row["repeat"] > 0] for value in (before, after)]
assert all(len(samples) >= gates["required_warm_samples_per_variant"] for samples in times)
medians = [float(np.median(samples)) for samples in times]
energy_error = abs(after["samples"][0]["energy"]-before["samples"][0]["energy"])
passed = (density_error <= gates["density_maxabs_difference"]
    and fock_error <= gates["fock_maxabs_tolerance_hartree"]
    and energy_error <= gates["total_energy_tolerance_hartree"])
payload = dict(scope=before["scope"], scientific_pass=passed,
    before_warm_seconds=times[0], after_warm_seconds=times[1],
    before_warm_median_seconds=medians[0], after_warm_median_seconds=medians[1],
    time_ratio_after_before=medians[1]/medians[0], density_maxabs_error=density_error,
    fock_maxabs_error_hartree=fock_error, energy_absolute_error_hartree=energy_error,
    before_xla_temp_bytes=before["xla_temp_bytes"], after_xla_temp_bytes=after["xla_temp_bytes"],
    before_process_peak_rss_bytes=before["process_peak_rss_bytes"],
    after_process_peak_rss_bytes=after["process_peak_rss_bytes"],
    source_sha256=[before["source_sha256"], after["source_sha256"]], gates=gates)
if args.native_patch:
    import hashlib
    payload['native_change'] = dict(patch=str(args.native_patch),
        patch_sha256=hashlib.sha256(args.native_patch.read_bytes()).hexdigest(),
        before_sha256=before['native_sha256'], after_sha256=after['native_sha256'])
args.output.write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if passed else 1)
