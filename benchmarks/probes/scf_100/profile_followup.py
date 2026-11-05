"""Reuse the stage profiler, adding only AO retention and RI whitening spans.

This is invasive diagnostic instrumentation, never a clean timing baseline.
Numerical production code, physical inputs and stopping rules are unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'scf_solver'))
import profile_stages

original_install = profile_stages.Stages.install


def install(self, representation):
    original_install(self, representation)
    from gradscf.scf.inputs import grid
    from gradscf.integrals.grids import blocks
    from gradscf.integrals.molecular.density_fitting import AuxiliaryMetricFactor

    self.patch(grid, 'prepare_scf_grid', 'grid.prepare_scf_grid')
    self.patch(blocks, '_materialize_scf_grid', 'grid.materialize_scf_grid')
    if representation == 'df':
        self.patch(AuxiliaryMetricFactor, 'whiten', 'integrals.whiten', aliases=False)


if __name__ == '__main__':
    profile_stages.Stages.install = install
    raise SystemExit(profile_stages.main())
