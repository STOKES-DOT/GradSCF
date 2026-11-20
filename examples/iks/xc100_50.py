"""Fifty real Luna/high proposals; all 100 XC100 targets are training targets.

The local SDK only designs formulas. A single allocated Slurm worker
does all numerical preparation, parameter fitting, AD and grid diagnostics.
This script makes real model calls and has no CLI. Resume the same directory;
changed numerical source/policy requires a new immutable campaign directory.
"""
from pathlib import Path

from gradscf.model.autoxc.workflows.xc100.distributed import design_campaign

HOST = 'jy_gpu_2'
REMOTE = '/home/user_yuanjiao/private/homefile/gradscf-xc100-pbe50-20261117'
OUTPUT = Path('artifacts/xc100-pbe-50')
OUTPUT.mkdir(parents=True, exist_ok=True)

brief = """Design an analytic XC energy-density CORRECTION to PBE. This is a
training-only campaign: all 100 published closed-shell targets enter the same
global coefficient fit and reward. No validation/test score or generalization
claim exists. You are the formula designer only: do not use tools, commands,
files, data access, source edits or numerical calculations. Return the Candidate
JSON schema. Python and GradSCF own all scoring and parameter fitting.

Prefer kind='energy'; kind='potential' defines an explicit local-potential
correction instead. Do not add another PBE baseline. Use up to four dimensionless
coefficients a,b,c,d. Initial values should be zero; bounds must be subsets of
[-0.2,0.2]. Coefficients must enter AFFINELY: no coefficient products, nonlinear
functions of coefficients, or coefficient denominators. Constants inside shape
functions are fixed design choices. Do not memorize target identities.

Features at the fixed target (atomic units):
n=sqrt(rho^2+eta^2), eta=1e-12 bohr^-3;
sigma=|grad rho|^2, grad=sqrt(sigma), lap=laplacian rho;
s2=sigma/[4*(3*pi^2)^(2/3)*n^(8/3)], s=sqrt(s2), u=s2/(1+s2);
q=lap/[4*(3*pi^2)^(2/3)*n^(5/3)], z=q/sqrt(1+q^2).
No tau or orbital-dependent feature is available in this local-WY campaign.
Favor smooth bounded u/z dependence and avoid singular low-density behavior.
The correction must vanish at sigma=lap=0 for all coefficient values; the
numerical protocol rejects uniform-limit violations. Spin transfer is untested.

Allowed formula syntax: +,-,*,/, fixed numeric powers with magnitude <=8,
sqrt,exp,log,log1p,tanh. Exponents must be numeric literals, e.g.
1.3333333333333333, not 4/3; no intermediate assignments or pi symbol.
The parser limits expressions to 120 nodes and depth 12. Mathematical domain
failures invalidate the entire candidate, not just a failed molecule.

For each fixed coefficient vector the evaluator builds F=T+Vext+J[target]+Vxc,
fills the occupied subspace once and computes W=Es-Tr[(F-T)Dtarget]. Reward is
the mean over ALL100 of (W-W_PBE)/(Ne*0.001 Ha). PBE reward=0. Higher reward is
better; this is not a self-consistent energy or a pointwise potential error.
The same globally fitted coefficients apply to all targets. Fitting uses
AD and bounded L-BFGS-B, not model-estimated scores. Density L2 and spectral
validity are separate diagnostics. Use previous measured reward, coefficients,
active bounds and aggregate diagnostics to revise your FORM, not the evaluator.
Each turn must propose one complete candidate, preferably a small justified
change from the incumbent while exploring different smooth functional shapes.
"""

result = design_campaign(HOST,REMOTE,output=OUTPUT,brief=brief,rounds=50)
print('Finished',len(result['records']),'rounds; training reward',result['incumbent']['reward'],flush=True)
