"""Collect small measurement records without changing their scientific status."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
args = parser.parse_args()
raw = args.report / "raw"
hashes, validation = {}, []
with (args.report / "measurements.jsonl").open("w") as output:
    for path in sorted(raw.glob("*.jsonl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.name] = digest
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                row["evidence"] = dict(file="raw/" + path.name,
                    sha256=digest, line=line_number)
                output.write(json.dumps(row, allow_nan=False) + "\n")

for path in sorted(raw.glob("*.xml")):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    counts = {key: sum(int(item.get(key, "0")) for item in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    exit_path = path.with_suffix(".exit")
    validation.append(dict(evidence="raw/" + path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **counts,
        seconds=sum(float(item.get("time", "0")) for item in suites),
        exit_code=int(exit_path.read_text().strip()) if exit_path.exists() else None,
        note="Per-run counts overlap; do not sum as unique coverage."))
for name, data in (("measurement_hashes.json", hashes),
                   ("validation_summary.json", validation)):
    (args.report / name).write_text(json.dumps(data, indent=2) + "\n")
print(f"Collected {len(hashes)} measurement files and {len(validation)} JUnit runs")
