"""Two live Codex proposals around a qualified XC100/PBE single-molecule target.

This script invokes the optional official Codex Python SDK using its existing
authentication. All molecular calculations use local CPU/float64 and GradSCF.
PySCF is used only by the independent baseline acceptance check, without SCF.
Raw data and numerical output stay outside version control. There is no CLI.
"""
import json
from pathlib import Path
import resource
import time

import jax

from gradscf.model import autoxc
from gradscf.model.autoxc.history import json_value, source_digest
from gradscf_tools.xc_design.molecule import XC100Case
from gradscf_tools.xc_design.checks import baseline_checks, candidate_checks

jax.config.update('jax_enable_x64', True)

DATA = Path('/Volumes/TF/GradSCF-data/XC100/gradscf-density-20261110')
OUTPUT = Path('artifacts/xc100-codex-single')
MODEL = 'gpt-5.6-luna'
ROUNDS = 2
OUTPUT.mkdir(parents=True, exist_ok=True)
proposal_directory = OUTPUT/'designer'
proposal_directory.mkdir(exist_ok=True)

started = time.perf_counter()
print('Preparing single-molecule PBE baseline', flush=True)
case = XC100Case(DATA, '010', floor=1e-12, block_size=256).prepare()
baseline = baseline_checks(case)
(OUTPUT/'baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
if not baseline['passed']:
    raise RuntimeError('PBE baseline failed independent acceptance; stop before any model call')
print('Baseline accepted; relative density L2 =', baseline['relative_density_l2'], flush=True)

brief = """A bounded, single-training-system smoke test of analytic XC design.
You are a formula designer only. Do not use tools, read files or datasets, run
commands, edit code, or perform the evaluation. The Python evaluator owns all
computation. Return one Candidate JSON object in the supplied schema.

The formula is a CORRECTION to a fixed PBE baseline, not the complete PBE model.
Prefer kind='energy' (energy density); kind='potential' is also supported.
Use at most two dimensionless coefficients named a and b, initial values zero,
with finite bounds inside [-0.2,0.2]. Coefficients must enter affinely: no products
of coefficients, nonlinear functions of coefficients, or coefficient denominators.
Propose a modest smooth correction that vanishes in the uniform-density limit.

Available fixed-target features, in atomic units:
n=sqrt(rho^2+eta^2), eta=1e-12 bohr^-3; sigma=|grad rho|^2;
grad=sqrt(sigma); lap=laplacian rho;
s^2=sigma/[4*(3*pi^2)^(2/3)*n^(8/3)]; u=s^2/(1+s^2);
q=lap/[4*(3*pi^2)^(2/3)*n^(5/3)]; z=q/sqrt(1+q^2).
Use n and bounded u/z to avoid tail singularities. Do not add a second PBE term.
Allowed syntax: arithmetic +,-,*,/, literal powers of magnitude <=8, and
sqrt, exp, log, log1p, tanh. Exponents must be numeric literals, for example
1.3333333333333333 rather than 4/3. Intermediate assignments and pi as a
symbol are unavailable. Formula values have energy-density units for energy
candidates, potential units for potential candidates.

The evaluator fits coefficients with AD on this one fixed target, without SCF.
Reward is DeltaW/(Ne*0.001 Ha), where W=Es-Tr[(F-T)Dtarget] and F contains XC
exactly once. Positive reward improves the finite-AO Wu-Yang objective over PBE.
It does not certify a unique exact potential, SCF accuracy or transferability.
In the next turn, use the measured reward/parameters to revise the proposal.
"""


class CheckedEvaluator(autoxc.Evaluator):
    def evaluate(self, candidate):
        start = time.perf_counter()
        print('Evaluating', candidate.formula, flush=True)
        result = super().evaluate(candidate)
        if result.valid:
            checks = candidate_checks(case.bind(candidate)[case.system_id], result.parameters)
            (OUTPUT/f'gradient-{candidate.digest[:12]}.json').write_text(json.dumps(checks, indent=2)+'\n')
            if not checks['passed']:
                raise ValueError('Molecular candidate AD/finite-difference check failed')
        print('Evaluation:', result.reward, 'valid:', result.valid,
              'seconds:', time.perf_counter()-start, flush=True)
        return result


evaluator = CheckedEvaluator(case.bind, sample_ids=(case.system_id,),
    fingerprint=case.fingerprint+';candidate-gradient-gate-v1', maxiter=30)
print('Opening Codex SDK thread:', MODEL, 'high', flush=True)
with autoxc.CodexDesigner(cwd=proposal_directory, model=MODEL, effort='high', timeout=300.) as designer:
    result = autoxc.run(designer, evaluator, directory=OUTPUT/'run', rounds=ROUNDS, context=brief)

report = dict(baseline=baseline, records=result['records'], incumbent=result['incumbent'],
    source_digest=source_digest(), case_fingerprint=case.fingerprint,
    archive_sha256=case.archive_hash, system_id=case.system_id,
    grid_points=case.grid.ngrids, grid_level=5, spherical_nao=case.target.overlap.shape[0],
    model=MODEL, effort='high', rounds=ROUNDS, wall_seconds=time.perf_counter()-started,
    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,
    backend=jax.default_backend(), float64=bool(jax.config.jax_enable_x64))
(OUTPUT/'acceptance.json').write_text(json.dumps(json_value(report), indent=2)+'\n')
print('Completed records:', len(result['records']), flush=True)
print('Incumbent:', result['incumbent'], flush=True)
print('Receipt:', OUTPUT/'acceptance.json', flush=True)
