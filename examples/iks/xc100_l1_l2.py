"""100 fresh potential proposals with outer L1/L2 scoring and strict inner gradients.

The local official SDK owns formula design; one Slurm GPU owns numerical work.
The completed 50-round experiment and its source receipts remain separate.
"""
from pathlib import Path

from gradscf.model.autoxc.workflows.xc100.distributed import design_campaign

HOST = 'jy_gpu_2'
REMOTE = '/home/user_yuanjiao/private/homefile/gradscf-xc100-l1l2-100-20261117'
OUTPUT = Path('artifacts/xc100-l1l2-100-20261117')

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

Two distinct objectives are used. The inner global coefficient fit maximizes
mean (W-W_PBE)/(Ne*0.001 Ha) over all 100 targets, where F=T+Vext+J[Dtarget]+Vxc,
W=Es-Tr[(F-T)Dtarget], and G=Tr[F Dtarget]-Es. Convergence requires the infinity
norm of the projected gradient of the minimized negative inner reward <=1e-6.
At a coefficient bound the raw gradient may be nonzero. The 500-iteration
ceiling is a safety guard: unresolved fits are invalid and cannot be accepted.

After fitting, the OUTER formula-selection reward includes actual induced
single-diagonalization densities. L1=integral |rho-rho_target| / integral
|rho_target|; Q=integral (rho-rho_target)^2 / integral rho_target^2. All terms
are averaged equally over all100. Fixed PBE dataset averages normalize them:
J=0.5*mean(G/Ne)/mean_PBE(G/Ne)+0.25*mean(L1)/mean_PBE(L1)
 +0.25*mean(Q)/mean_PBE(Q). Outer reward=1-J; PBE reward=0. Larger is better.
Formula selection uses THIS composite reward; inner coefficients still use WY.
These are training-only fixed-target diagnostics, not candidate SCF results.
No energy-to-density AD is needed. Coefficient AD remains active. Exact L1
and squared L2 share one blocked grid pass and the already-solved density.

Use measured reward, fitted coefficients, active bounds and aggregate density
diagnostics to revise the next FORM. Prefer small justified changes from the
incumbent while exploring different smooth shapes. Keep each formula and its
physical rationale explicit. Old energy/Laplacian experiments are not part of
this fresh 100-round potential-only history.

Favor different smooth shapes over repeatedly rescaling an unchanged basis. Explain which measured outer component motivates the change.
"""

result = design_campaign(HOST,REMOTE,output=OUTPUT,brief=brief,rounds=100)
print('Finished',len(result['records']),'rounds; composite training reward',
      result['incumbent']['reward'],flush=True)
