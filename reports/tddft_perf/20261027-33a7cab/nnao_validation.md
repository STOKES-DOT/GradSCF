# Actual NNAO validation scope

Two H2 geometries, 0.74/0.91 Angstrom; random MACE fixture, no trained weights.
Each has 24 primitive AOs, 8 contracted AOs and 36 explicit auxiliary AOs.
The production native RI projection and coefficient/log-exponent products are
used; no primitive-four-index workflow is introduced by the fixture.
Projection JVP/VJP and two FD step sizes passed for both geometries in the
focused regression XML. Numerical basis/SCF arrays are float64; original model
parameter dtypes were not captured separately.

Full DF-RHF variational-energy loss/gradient/JointAdam/checkpoint path:
first measured step 19.400778 s;
three updated-parameter warm steps median 0.015343 s.
Initial two-geometry mean energy objective -1.111120032002 Ha;
post-four-update re-evaluation -1.111183325089 Ha.
This objective is not an FCI MSE target. Finite gradients, frozen-mask equality,
optimizer state and checkpoint roundtrip were checked; loss decrease alone is
not the validation criterion. Metric/geometry contexts stay fixed separately
per geometry while coefficient/exponent values are rebound each update.

Model-weight VJP, basis pullback and implicit solve are exercised in the full
training step, but their pure kernel timings are not all separately isolated.
No NNAO excitation training, force-loss mixed derivatives or real-checkpoint
accuracy is claimed. See runs/existing-ad-contracts-neural-nnao-cpu1/nnao-step.json.
