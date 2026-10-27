# Exact local experiments (unapplied production patches)

1. **packed-public**: replace the private TD s4 J/K contraction with the public
   bounded implementation. Batched non-symmetric densities remain unchanged.
2. **df-demand**: TDA prepares J-VO, K-OO and K-VV only. The required branch is
   known statically. The standalone factory is in the instrumentation patch;
   propagating a new production option is intentionally outside this probe.
3. **df-share**: full TDDFT reuses J-OV/VO for K only when the J/K source object
   is identical. Distinct/non-symmetric raw factors keep their original meaning.

No patch is applied to `src/`. The harness installs each experiment only in a
fresh child process, with explicit configuration and separate output folders.
The TDA demand and full sharing runs change separate factors and are not
reported as a combined optimization. `experiments.py` is the executable
reference for the diagnostic implementation; full integration is not claimed.
