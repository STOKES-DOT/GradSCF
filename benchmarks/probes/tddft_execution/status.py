"""Run acceptance uses original-operator residuals, not exit codes or energies."""
from __future__ import annotations
import math


def summarize(process, records, *, residuals=None, tolerance=1e-8,
              supersedes_run=None, residual_source=None):
    completed=[x for x in records if x.get('stage')=='end_to_end']
    per_root=None if residuals is None else [math.isfinite(x) and x<=tolerance for x in residuals]
    scf=[x.get('converged') for x in records if x.get('stage')=='reference_metadata']
    public=[x.get('public_converged',x.get('status','ok')=='ok') for x in completed]
    scientific=(None if per_root is None else bool(per_root and all(per_root)
                 and scf and all(x is True for x in scf) and public and all(public)))
    return dict(process_status=process.get('status','unknown'),
                scientific_pass=scientific,response_converged_per_root=per_root,
                residual_per_root=residuals,residual_source=residual_source,
                accepted_for_comparison=bool(process.get('status')=='ok' and completed and scientific is True),
                timeout_scope=process.get('timeout_scope','whole_repeated_process' if process.get('status')=='timeout' else None),
                completed_solves=len(completed),scf_converged=scf,supersedes_run=supersedes_run)
