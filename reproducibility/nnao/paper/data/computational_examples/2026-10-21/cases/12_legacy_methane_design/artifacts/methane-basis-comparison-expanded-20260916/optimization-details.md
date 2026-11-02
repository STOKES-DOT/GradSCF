# Expanded primitive optimization

C: fixed six-primitive core, valence/polarization 4s4p2d -> 6s6p3d.
H: 3s1p -> 6s2p; the p contraction is now trainable. No added contracted
radial functions; CH4 stays at 26 spherical AOs, while primitive AOs grow
from 56 to 93. Primitive ERI storage grows by (93/56)^4 = 7.606.

Old exponents are preserved, with added tight/diffuse endpoints (2*max,
min/2), then a geometric midpoint in the largest logarithmic gap where
needed. All exponents stay fixed. Core coefficients stay fixed; MACE
predicts complete valence/polarization vectors, including new H-p vectors.
The output head is (natom,3,6). This is a custom unoptimized exponent grid.

Initialization embeds the preceding optimized coefficients with new entries
zero. The trainable output bias is initialized; the MACE backbone uses seed
0 and zero head kernels. This transfers coefficients, not a whole trained
network checkpoint. No fixed empirical vector is added in forward.

- Initial RHF: -40.172528224491 Ha.
- Final RHF: -40.186970229395 Ha.
- Improvement relative to preceding optimized basis: 14.442005 mHa.
- Accepted steps: 463; objective evaluations: 502.
- Optimizer success: True; termination: CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL.
- Maximum final parameter gradient: 4.62873e-07 (gtol=5e-7).
- Network direction FD error, initial/final: 4.23132e-11 / 2.78459e-10, step=1e-4.
- Training elapsed: 328.555 s on CPU; hardware is recorded in the independent comparison JSON.

Local regression: 82 passed, 6 optional dependency skips. Remote NNAO suite:
80 passed (143.88 s), including actual MACE/JIT. All 710 recorded source and
resource hashes match local files. The wheel contains the expanded basis
and head code. The initial 160-step trial is preserved in
artifacts/methane-szp663-160steps/; the repeated first 160 energies match
exactly in the final run.

The derivative test uses the stationary first-order RHF Lagrangian with
Pulay terms and fixed native primitive integrals. Higher-order derivatives
are not tested here. Neither optimized basis has been validated for transfer
to other molecules/geometries. All benchmark results remain finite-basis RHF.
