"""Real H3 doublet UHF/ROHF -> UCISD/UCCSD and fixed-MO integral response.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/cc/open_shell_ground.py
No PySCF calls are made. Molecular integral construction uses the configured
GradSCF CPU backend. Distances are Angstrom and all energies are Hartree.
"""
import json
import platform
import time
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gradscf import cc, ci, gto, scf
from gradscf.scf.reference import unrestricted_reference_from_source


def main():
    started = time.perf_counter()
    mol = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9", basis="sto-3g", spin=1)
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
              "dtype": "float64", "molecule": "H3 doublet", "basis": "STO-3G",
              "z_angstrom": [0., .85, 1.9], "integral_backend": "cpu",
              "energy_unit": "Hartree", "scf_conv_tol": 1e-11,
              "cc_residual_tol": 1e-11, "ci_conv_tol": 1e-11}
    for kind in ("UHF", "ROHF"):
        mf = getattr(scf, kind)(mol, conv_tol=1e-11, max_cycle=150).run()
        cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
        coupled = cc.UCCSD(mf, conv_tol=cfg.conv_tol, residual_tol=cfg.residual_tol).run()
        variational = ci.UCISD(mf, conv_tol=1e-11).run()
        full = ci.UCI(mf, max_excitation=3, conv_tol=1e-11).run()
        if not all((mf.converged, coupled.converged, variational.converged, full.converged)):
            raise RuntimeError(f"{kind} post-HF example did not converge")
        ref = unrestricted_reference_from_source(mf)
        direction = jnp.diag(jnp.array([.2, -.1, .07]))
        def energy(x):
            return cc.run_ucc((ref.h1[0]+x*direction, ref.h1[1]), ref.eri,
                              nocc=ref.nocc, nuclear_repulsion=ref.nuclear_repulsion,
                              config=cfg).total_energy
        analytic = jax.jit(jax.grad(energy))(0.)
        step = 1e-4
        finite = (energy(step)-energy(-step))/(2*step)
        report[kind] = {"hf": float(mf.e_tot), "ucisd": float(variational.e_tot),
                        "uccsd": float(coupled.e_tot), "full_ci": float(full.e_tot),
                        "cc_iterations": int(coupled.result.iterations),
                        "cc_residual_norm": float(coupled.result.residual_norm),
                        "ci_residual_norm": float(variational.result.residual_norms[0]),
                        "t1_shapes": [list(t.shape) for t in coupled.t1],
                        "t2_shapes": [list(t.shape) for t in coupled.t2],
                        "fixed_mo_halpha_derivative": float(analytic),
                        "finite_difference": float(finite), "step": step,
                        "derivative_error": float(jnp.abs(analytic-finite))}
    report["elapsed_seconds"] = time.perf_counter()-started
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
