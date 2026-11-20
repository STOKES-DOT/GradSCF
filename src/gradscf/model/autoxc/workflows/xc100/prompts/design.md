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

## Available density inputs

The active evaluator registry controls which names are available. The
standalone density worker provides these eight fields. The density is
the input density at scoring time and the current density at SCF runtime.

| Name | Definition in atomic units |
| --- | --- |
| `n` | $\sqrt{\rho^2+10^{-24}}$ |
| `sigma`, `grad` | $\lvert\nabla\rho\rvert^2$ and its square root |
| `s`, `u` | $s=\text{grad}/[2(3\pi^2)^{1/3}n^{4/3}]$, $u=s^2/(1+s^2)$ |
| `lap`, `q`, `z` | $\nabla^2\rho$, $q=\text{lap}/[4(3\pi^2)^{2/3}n^{5/3}]$, $z=q/\sqrt{1+q^2}$ |

The expression language accepts `+`, `-`, `*`, `/`, `**`,
`sqrt`, `exp`, `log`, `log1p`, and `tanh`. Use finite numeric
literals. Literal exponent magnitude is at most eight. The combined
limits are 16,000 characters and 2,048 expression nodes, with a depth
limit of 64 per formula or definition. Use the names and arithmetic
provided by the registry and the proposal schema.

## Orbital inputs to enable with the candidate state provider

The following groups define the orbital-aware experiment. An input
becomes available for scoring when the evaluator registers it and
supplies its runtime and gradient rules. With the current standalone
density worker, use the density-only experiment as a control.

| Descriptor family | Construction from the candidate's state |
| --- | --- |
| Occupied orbital shapes | Supply $\tau_{\mathrm{occ}}=\tfrac12\sum_i f_i\lvert\nabla\phi_i\rvert^2$ and an iso-orbital indicator derived from that kinetic density and its corresponding occupied density. |
| Exchange hole | Use the spin-resolved occupied density matrices to build the exchange hole. Supply its Coulomb potential or the short- and long-range Coulomb contractions in a declared gauge. |
| Virtual spectral control | Sum the virtual density with Lorentzian energy weights centered at the candidate's HOMO–LUMO midpoint. Use separate window widths of 0.2 and 1.0 Ha for comparison with the earlier spectral construction. |
| Occupied–virtual response | Use transition densities and gaps to project $\chi_0(i\omega)$ onto a specified perturbation. One option is $P_{\omega,\mu}(\mathbf r)=-\langle K_{\mu,\mathbf r},\chi_0(i\omega)K_{\mu,\mathbf r}\rangle$, with $K_{\mu,\mathbf r}$ a range-separated Coulomb perturbation centered at $\mathbf r$. |
| Perturbative correlation | Supply a defined contraction of PT2 amplitudes or a pair-density correction, with the integral convention, gaps, regularization, and localization rule recorded. |

For the orbital-aware iKS protocol, keep the target Hartree operator
and any explicit target-density inputs fixed. Build orbital descriptors
from the trial potential's own orbitals and update that state to
convergence for each parameter evaluation. Define the orbital density
used in derived observables such as the iso-orbital indicator. The
inner gradient includes the response of the converged orbital state.
After scoring, assess independent SCF by updating the density, Hartree
operator, and orbital inputs together from the declared regular initial
guess. Record SCF convergence and density error as separate outcomes.

## Inner fit and outer search

**Inner fit.** Hold the potential form and the scoring protocol fixed.
Use AD and bounded optimization to maximize Wu–Yang over the
declared shared parameters. The projected-gradient infinity norm must
reach $10^{-6}$. Warm-start from a compatible fitted parent when
possible. When a new term has a parameter value that recovers the
parent, use that as a matched control. Record the parameter values,
active bounds, numerical failures, and convergence.

**Outer search.** Use one structure change per scientific comparison.
Begin with a density-only control, then compare occupied exchange
information, virtual spectral information, and explicit occupied–virtual
response as the evaluator enables those inputs. Test short-/long-range
constructions and combinations with a matched refit. An available
perturbative contraction can supply a separate correlation branch.

Use the controller's measured rewards and action feedback. Interpret a
structural outcome with the fitted score of its matched parent and the
density diagnostics. Record repeated forms and fit-only changes
independently of new physical forms. A controller request for `refit`
is a numerical control of the same structure. For structural actions,
register the coefficient of a proposed physical term once and let the
inner fit determine its value. Explain the mechanism, physical limit,
and matched removal or replacement in `hypothesis`.
