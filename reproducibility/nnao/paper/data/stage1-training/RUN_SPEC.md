# GMTKN55 first shared NNAO training batch

This file is an English translation of the imported source record, not a new run or result. Dates, status statements, and numerical values refer to that source record.

2026-10-07 user authorization: set the implicit linear-solve tolerance for this round to `1e-8`, then train for **1000 epochs** after validation is complete.

## Data and model

Use the fixed GMTKN55 v2 small-sample selection: 24 CHNOF, neutral, spin=0 structures, comprising 16 core selections, 6 extended selections, and 2 associated monomers. All are training samples in this round; there is no independent test set, and generalization is not assessed from them.

One shared MACE processes all structures with the element vocabulary `(H,C,N,O,F)`. Configuration: channels=8, 2 interactions, max_ell=1, correlation=2, zero_init=True, seed=0. Basis `szp663_direct/core6`, spherical AOs, no ECP; geometry, primitive exponents, and inner-shell parameters are fixed.

## Objective and updates

\[
L(\theta)=\frac{1}{24}\sum_i E_{\mathrm{RHF,DF}}(R_i,c_\theta(R_i)).
\]

Each structure first converges SCF, followed by implicit backward through the density fixed point. Each epoch accumulates 24 gradients at the same parameter vector, then performs one Adam update. Learning rate `1e-4`, global gradient clip=1. Reference energies do not enter the loss; no additional regularization term is used.

Forward SCF retains `max_cycle=150`, energy tolerance `1e-12 Ha`, density tolerance `1e-10`, and gradient tolerance `1e-9`. Implicit settings are `tolerance=1e-8`, max_iter=100, restart=40, regularization=0.

## Startup requirements and checkpoints

Validate the previously failed methane dimer first, then rerun all 24 records: directional finite differences, 5-step Adam, and SCF/basis diagnostics must all pass before training. Any invalid structure during training stops that step and retains the last valid checkpoint; samples are not silently discarded.

`last.npz` and `best.npz` store parameters, the complete Adam state, epoch, and loss/gradient evaluated at those parameters, allowing strict continuation. Each epoch records the mean energy and per-structure diagnostics. Native fixed integrals use the existing CPU DF cache; GPU executes contraction, SCF, and AD. No full ERI is stored.

Slurm uses the home partition, one V100, 16 CPU, and 64 GiB. The validation time limit is 1 hour and the training time limit is 4 hours. Run records store actual hardware, software versions, and source/data/cache hashes; execution status is in the corresponding logs, summary, and status files.
