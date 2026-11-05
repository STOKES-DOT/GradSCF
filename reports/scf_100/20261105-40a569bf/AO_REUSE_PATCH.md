# T2.3: actual production AO reuse

The 880-AO diagnostic showed a streamed provider and about 27.4 s for the
initial Fock, 159.7 s for the fixed five-cycle solver, and 173.5 s preparing
inputs, including 105.9 s for RI factors. These synchronized inclusive spans
are not additive children and are not formal clean timings.

The old planner compared full-grid primitive scratch against the budget.
Rejecting that one-shot evaluation did not mean retained AO planes could not
fit. The streamed provider then recomputed Gaussian values each SCF pass.

The new production path accounts for inputs, forward state matrices and known
operator workspace; prepares bounded Gaussian blocks once; and reuses exact
grid-leading planes through bounded gathers. The derived reducer is local to
the solve and rebuilt for new physical inputs. Explicit streaming, inadequate
budget, unknown operators/devices and existing external/global callbacks keep
their old paths. No integral engine, grid approximation, orbital cutoff,
stop-gradient or public switch is introduced.

Validation before full timing:

- Execution RED: 3 failures, 4 existing guards passed, 5.85 s.
- First AD/value GREEN: 7 passes, 8.91 s.
- Workspace RED: 6 failures, 1.30 s.
- Budgeted candidate and existing regressions: 96 passes, 152.79 s.
- Exact-buffer refactor: 13 passes, 10.50 s, including JVP/VJP, dense HVP,
  two-step FD, changing inputs, ownership and fallback checks.

The same-input 880-AO XC prototype reduced warm action from 25.1830 s to
12.0135 s, with zero E/V differences. Its 106.7303 s preparation is separately
charged. Factor capacity was reserved but factors were not physically held;
this is an operator experiment, not a converged molecular energy.

The optimized complete RKS/PBE/RI-J cold call returned in 577.9149 s versus
792.4915 s for the parent. Both used 23 cycles with exactly the same recorded
energy and final gradient. All preparation is included. Three optimized warm
repetitions remain in progress at this checkpoint; final statistics, memory
and the remaining matrix are open. No universal or GPU speedup is claimed.
