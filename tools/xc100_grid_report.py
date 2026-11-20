"""Render and validate saved grid-block benchmark results inside the allocation."""
import argparse
import json
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--root')
    inputs.add_argument('--summary', type=Path, help='Only redraw already validated scalar measurements')
    args = parser.parse_args()
    from gradscf.model.autoxc.workflows.xc100.grid_report import build_report, render_plot
    if args.summary:
        data = json.loads(args.summary.read_text())
        render_plot(data['rows'], data['sweep'], args.summary.parent)
        print('Rendered', len(data['rows']), 'saved measurements')
    else:
        rows = build_report(args.root)
        print('Validated', sum(row['status']=='ok' for row in rows), 'of', len(rows), 'points')
