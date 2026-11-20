"""Fifty analytic-potential proposals, starting from fixed PBE references.

Run locally: the official Codex SDK designs formulas, while the one allocated
Slurm worker integrates potentials, fits coefficients and evaluates all targets.
Resume this same output directory to recover a saved proposal/thread. This is
a training-only experiment on all 100 XC100 targets, with no holdout claims.
"""
from pathlib import Path

from gradscf.model.autoxc.workflows.xc100.distributed import design_campaign


HOST = 'jy_gpu_2'
REMOTE = '/home/user_yuanjiao/private/homefile/gradscf-xc100-potential-20261117'
OUTPUT = Path('artifacts/xc100-potential-50-20261117')

brief = """Design an analytic LOCAL XC POTENTIAL CORRECTION to the fixed,
handwritten PBE/PW92 reference. Use kind='potential' exclusively. Do not return
an energy density, differentiate a parent energy, or add PBE a second time.
The output correction is in Ha. All 100 XC100 targets are fitting/reward data;
there is no validation/test score and no generalization claim.

You design the FORM only. Do not use tools, commands, files, datasets, source
edits or numerical calculations. Return one complete Candidate JSON object.
Python/GradSCF owns integration, coefficient fitting, diagonalization and reward.
Use up to four coefficients a,b,c,d, initialized to zero, with bounds contained
in [-0.2,0.2]. Coefficients must enter AFFINELY: no coefficient products,
denominators or nonlinear coefficient functions. Fixed shape constants are
your design choices. One globally fitted coefficient vector serves all targets.

Available fixed-target features, in atomic units:
n=sqrt(rho^2+eta^2), eta=1e-12 bohr^-3;
sigma=|grad rho|^2; grad=sqrt(sigma);
s2=sigma/[4*(3*pi^2)^(2/3)*n^(8/3)]; s=sqrt(s2); u=s2/(1+s2).
Only n,sigma,grad,s,u are allowed. No lap,q,z,tau or orbital information.
The correction must vanish at sigma=0 for every allowed coefficient vector.
Favor smooth bounded shapes and regular low-density behavior. A useful starting
scale is n**0.3333333333333333 times a smooth function of u that vanishes at u=0.
This is a scalar potential model using density/gradient descriptors; it is not
claimed to be the variational derivative of a GGA energy or to yield forces.

Allowed syntax: +,-,*,/, numeric literal powers of magnitude <=8,
sqrt,exp,log,log1p,tanh. Powers must be numeric literals, e.g.
0.3333333333333333, not 1/3. No assignments or pi symbol. Maximum 120 syntax
nodes and depth 12. Any nonfinite field, spectral failure or omitted target
invalidates the candidate. Do not memorize target identities. Spin transfer
and self-consistent transfer are untested.

For each fixed coefficient vector, F=T+Vext+J[Dtarget]+Vxc is diagonalized
once. W=Es-Tr[(F-T)Dtarget]. Reward is the mean over all 100 targets of
(W-W_PBE)/(Ne*0.001 Ha), with PBE reward zero. Larger is better. This is a
finite-AO Wu-Yang variational score, not a self-consistent energy or an error
against a precomputed exact potential. Density relative L2 is a separate
diagnostic. Ordinary coefficient AD and bounded L-BFGS-B fit parameters.

Use measured reward, fitted coefficients, active bounds and aggregate density
diagnostics to revise the next FORM. Prefer small justified changes from the
incumbent while exploring different smooth shapes. Keep each formula and its
physical rationale explicit. Old energy/Laplacian experiments are not part of
this fresh 50-round potential-only history.
"""

result = design_campaign(HOST, REMOTE, output=OUTPUT, brief=brief, rounds=50)
print('Finished', len(result['records']), 'rounds; training reward',
      result['incumbent']['reward'], flush=True)
