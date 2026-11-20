"""Open analytic-potential search under the density-only Wu–Yang objective.

Formula design runs locally through the official Codex SDK. One allocated
Slurm worker owns all integration, reference orbitals, fitting and diagnostics.
All 100 XC100 samples are training data. This file has no command-line parser.
"""
from pathlib import Path

from gradscf.model.autoxc.workflows.xc100.distributed import design_campaign


HOST = 'jy_gpu_2'
REMOTE = '/home/user_yuanjiao/private/homefile/gradscf-iks-open-20261117-r2'
OUTPUT = Path('artifacts/iks-open-100-20261117-r2')

brief = """Design explicit analytic LOCAL XC POTENTIAL corrections to the fixed
handwritten PBE/PW92 baseline. Use kind='potential'. Output delta_vxc in Ha;
the numerical assembly adds PBE once. No target energy, exact potential or
energy functional is required. All 100 XC100 systems participate in fitting
and feedback; results are training performance, not holdout accuracy.

You choose the FORM and its inputs. Python owns numerical evaluation and
coefficient fitting. Do not use tools, files, datasets, source changes or
model-estimated scores. Return the complete JSON Candidate schema, including
definitions (an empty array if none). Definitions are ordered objects
{name,expression}, and may only refer to earlier definitions. Return one
formula that uses those intermediates. Parameter names must be distinct from
physical inputs. Every parameter has a finite initial value and finite lower
and upper bounds containing it. There is NO four-term or four-parameter limit,
no fixed density prefactor, no required parameter linearity, and no forced
gradient-only shape or uniform-limit deformation.

Available scalar fields, in atomic units:
n=sqrt(rho_target^2+1e-24); sigma=|grad rho_target|^2; grad=sqrt(sigma);
s2=sigma/[4*(3*pi^2)^(2/3)*n^(8/3)]; s=sqrt(s2); u=s2/(1+s2);
lap=laplacian rho_target; q=lap/[4*(3*pi^2)^(2/3)*n^(5/3)]; z=q/sqrt(1+q^2);
tau is the target-density-matrix kinetic energy density; alpha=(tau-tauW)/tauUEG,
tauW=sigma/(8*n), tauUEG=0.3*(3*pi^2)^(2/3)*n^(5/3).
rho_occ and tau_occ instead use the declared PBE@fixed-target-density reference
orbitals. They may differ from target fields. virtual_0 and virtual_1 are
energy-filtered reference virtual densities, using widths eta=0.2 and 1.0 Ha:
sum_a 2*|phi_a(r)|^2*eta^2/[(epsilon_a-mu)^2+eta^2], where mu is the reference
midgap. They are finite-basis descriptors; basis/window transfer is untested.
References and descriptors stay fixed during each coefficient fit.

Allowed scalar operations: +,-,*,/,** and sqrt,exp,log,log1p,tanh.
Use Python ** for powers, NEVER ^. Dynamic exponents and nonlinear parameters
are allowed. Fixed literal exponent magnitude is limited to eight. The parser
budgets all definitions/formula together at 16000 characters and 2048 nodes,
with depth at most64 per expression. There are no assignments inside expression
strings, array indexing, attribute access or executable Python. Keep domains
and parameter bounds finite on low-density grids. Any nonfinite field/gradient,
failed spectrum or omitted system invalidates the entire candidate.

For each parameter vector, assemble the candidate local AO potential, solve
the frozen one-electron Fock equation once and evaluate
W=Es-Tr[(F-T)Dtarget], F=T+Vext+J[Dtarget]+Vxc. Es is the occupied spectral sum
computed from the trial potential, not an external energy label. The PRIMARY
reward is mean[(W-W_PBE)/(Ne*0.001 Ha)] over ALL100. PBE reward=0; larger is
better. This strictly follows the iKS variational objective; it is not an XC
energy error or a certificate of unique continuum/pointwise potential accuracy.

The same shared parameters are fitted globally. Inner success requires
projected-gradient infinity norm <=1e-6; raw gradients can be nonzero at active
bounds. A 500-iteration ceiling and 1800-second boundary-checked fitting budget
are failure guards. Unresolved fits cannot be accepted. Affine parameter models
can cache AO directions; general nonlinear models are equally valid but repeat
integration and its shared VJP as parameters change. Runtime/complexity are
recorded, not secretly added to the scientific score.

Outer acceptance is WY-first: a reward increase above1e-8 is accepted. Within
that tolerance, a candidate can only replace the incumbent if WY does not
decrease AND its normalized L1+L2-squared density diagnostic improves. L1 is
integral |delta rho| / integral |rho_target|; L2-squared is integral delta rho^2
/ integral rho_target^2. Frozen PBE means normalize the secondary components.
Both are reported independently; they do not replace WY in the inner objective.

Use measured feedback to propose different physically motivated forms, input
couplings, smooth transitions or response scales. Avoid endlessly rescaling
one unchanged basis. Explain the tested hypothesis and which observed residual
motivates it. Do not call an idea successful until Python returns a valid score.
All formulas, parameters, failures and scores are retained in this fresh history.

Infrastructure correction: the two saved opening proposals are being replayed
without additional model calls after a fixed-reference symmetry tolerance was
repaired. The previous rejection occurred before candidate fitting; it is not
evidence against orbital inputs or either formula. All scores are recomputed
with the corrected source; use those measured results for subsequent designs.
"""

result = design_campaign(HOST,REMOTE,output=OUTPUT,brief=brief,rounds=100)
print('Finished',len(result['records']),'rounds; WY training reward',
      result['incumbent']['reward'],flush=True)
