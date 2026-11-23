# Design Prompt · XC100 local XC potential search

## Experiment and comparison

Design a standalone local XC potential for closed-shell XC100 molecules.
PBE/PW92 is the round-zero scoring baseline. Each trial reports the
same normalized Wu–Yang improvement and density diagnostics on the
declared XC100 batch. Use the feedback from the current campaign for
comparisons and report the provenance of any prior control.

For a density-only trial, the evaluator builds the Fock matrix at the
target density matrix $D_{\mathrm t}$ and computes $W_m$.

$$
F_\theta=T+V_{\mathrm{ext}}+J[D_{\mathrm t}]
 +V_{\mathrm{xc},\theta}[D_{\mathrm t}],
\qquad
W_m=\sum_i f_i\varepsilon_i(F_\theta)
     -\operatorname{Tr}[(F_\theta-T)D_{\mathrm t}].
$$

$V_{\mathrm{xc},\theta}$ is the AO projection of the entire submitted
potential. The normalized score compares each molecule with PBE.

$$
R_{\mathrm{WY}} = \operatorname{mean}_m
\frac{W_m-W_m^{\mathrm{PBE}}}{N_{e,m}(10^{-3}\,\mathrm{Ha})}.
$$

The target density supplies the fitting data. Report Wu–Yang, the mean
relative density L1 and L2 errors, and fit convergence separately.
Invalid samples invalidate the aggregate.

- In the Wu–Yang-first mode, use the normalized Wu–Yang improvement first.
  A nonnegative gain within the declared tolerance requires improvement
  in the normalized L1 and squared L2 secondary diagnostic.
- In `open_xc` mode, use the measured joint reward for structural selection.
  It equally weights normalized Wu–Yang improvement and reduction in the
  per-electron absolute XC energy error against `Exc_TZ`. The inner fit
  still maximizes Wu–Yang. The energy is evaluated once after fitting as
  $E_Q[\rho_\mathrm{out}]=\int d\mathbf r\,\rho_\mathrm{out}(\mathbf r)
  \int_0^1 v_\mathrm{xc}[t\rho_\mathrm{out}](\mathbf r)\,dt$.
  Submit the complete potential in a gauge with a vanishing vacuum limit.
  Keep its density path finite and avoid freely fitted constant shifts.

## Constructing density features

The evaluator supplies the spin-summed regularized density `n`, the squared
gradient magnitude `sigma`, the gradient magnitude `grad`, and the Laplacian
`lap`. Quantities are in atomic units. Declare custom features as named
entries in `definitions`, ordered by dependency. Definitions can be used
by later definitions and by the potential. The conventional `s`, `u`,
`q`, and `z` shorthands are accepted for existing analytic proposals.

The scalar expression language accepts `+`, `-`, `*`, `/`, `**`,
`sqrt`, `exp`, `log`, `log1p`, and `tanh`. The pair operators let you
design a kernel from the same scalar expressions.

| Operation | Grid value at the target point |
| --- | --- |
| `pair_integral(field, kernel)` | Integrate the source field times the kernel over all source points |
| `pair_average(field, kernel)` | Divide the integral by the integral of the kernel |
| `pair_moment(field, kernel, rank)` | Integrate the weighted displacement for literal rank 1 or its traceless second moment for rank 2, then return the squared norm |

In a kernel, a field name denotes its value at the target point. Prefix a
field or an earlier definition with `source_` to read its value at the source
point. `r2` is the squared displacement in bohr squared. Parameters are
shared between source and target and use their declared names. For example,
`pair_average(n, exp(-a*r2*(1+b*source_n)))` lets the source density change the
range. Declare `a` and `b` in `parameters` with appropriate units and bounds.
Return finite values on the scoring grid, including when the kernel is
normalized. The evaluator computes the result for each fitted parameter
point when the convolution has adjustable parameters.

Literal exponent magnitude is at most eight. The combined limits are
16,000 characters and 2,048 expression nodes, with a depth limit of 64 per
formula or definition.

## Inner fit and outer search

**Inner fit.** Hold the potential form and the scoring protocol fixed.
Use AD and bounded optimization to maximize Wu–Yang over the
declared shared parameters. The projected-gradient infinity norm must
reach $10^{-6}$. Warm-start from a compatible fitted parent when
possible. When a new term has a parameter value that recovers the
parent, use that as a matched control. Record the parameter values,
active bounds, numerical failures, and convergence.

**Outer search.** Use one structure change per scientific comparison.
Begin with a pointwise control. Test changes to the source field,
spatial kernel, angular moment, or the way the resulting features are
combined into the potential. Use matched refits for comparisons.

Use the controller's measured rewards and action feedback. Interpret a
structural outcome with the fitted score of its matched parent and the
density diagnostics. Record repeated forms and fit-only changes
independently of new physical forms. A controller request for `refit`
is a numerical control of the same structure. For structural actions,
register the coefficient of a proposed physical term once and let the
inner fit determine its value. Explain the mechanism, physical limit,
and matched removal or replacement in `hypothesis`.
