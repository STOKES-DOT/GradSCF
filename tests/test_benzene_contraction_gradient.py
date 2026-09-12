from pathlib import Path
import runpy

import jax.numpy as jnp
import numpy as np


def test_stationary_coefficient_gradient_includes_overlap_response():
    module = runpy.run_path(str(Path("tools/optimize_benzene_contractions.py")))
    experiment = module["ContractionExperiment"]("h2")
    x = experiment.x0
    initial = experiment.evaluate(x)
    assert initial["primitive_energy_error"] < 1e-10
    assert len(initial["gradient"]) == 1
    fd = module["check_gradient"](experiment, x, initial)
    assert fd["max_absolute_error"] < 1e-7
    trial = x-0.1*np.asarray(initial["gradient"])
    assert experiment.evaluate(trial)["energy_hartree"] < initial["energy_hartree"]
    np.testing.assert_allclose(experiment.transform(jnp.asarray(x)).T@experiment.ps@
                               experiment.transform(jnp.asarray(x)),
                               experiment.plan.evaluate("overlap", experiment.params), atol=1e-12)
