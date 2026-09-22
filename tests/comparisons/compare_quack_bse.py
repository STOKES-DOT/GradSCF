"""Opt-in static TDA/full-BSE oracle against unmodified pinned QuAcK Fortran.

Run with PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python tests/comparisons/compare_quack_bse.py
Missing pinned sources are downloaded into the temporary cache. Downloads/compilation
are intentionally outside normal pytest. Requires gfortran and NumPy/JAX.

QuAcK uses physicists ERI[p,r,q,s]=(pq|rs), and XpY[mode,transition].
We independently solve full direct RPA (not TDA screening): A=D+2K, B=2K,
sqrt(D)(D+4K)sqrt(D) U=U Omega**2; (X+Y)=sqrt(D) U / sqrt(Omega).
rho[p,q,s]=sum_ia (pq|ia)(X+Y)[ia,s]. At eta=0, lambda=1,
QuAcK KA=4 sum_s rho[i,j,s]rho[a,b,s]/Omega[s] is ONLY the screening
correction. Actual Fortran phRLR_A + KA is the complete singlet/triplet A.
Likewise phRLR_B + KB provides B. Full-BSE reference eigenvalues use NumPy
on the doubled matrix of these Fortran blocks, not a QuAcK eigensolver.
All energies and kernel elements are in Hartree. This synthetic factorized
model validates the static kernel conventions, not molecular GW accuracy.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

COMMIT = "2236bfcda24ff0971358f5107636b506dd201bb1"
SOURCES = {
    "src/GW/RGW_phBSE_static_kernel_B.f90": "b6e7340f8855165c37ea552509337c28b82873018577bdc8ea6a9faeb425d14f",
    "src/LR/phRLR_B.f90": "7efa5ba4d442f7869d012a5d536fd686ab1d660e4bdf8451169c43d13cf553ea",
    "src/GW/RGW_phBSE_static_kernel_A.f90": "4f499e3825c81caa2b50f3327e0df3d529e3a49551e64b46c27fb0772f8fe577",
    "src/LR/phRLR_A.f90": "c94b007d67b30ba3b25b7d915993b5178e81798eb1c88c0ea585af0fd603ac5e",
    "src/GW/RGW_excitation_density.f90": "9acca4e2fe6ff1d1f17c2d069f92aaea9d930fa422f97e1c2b89bbd82bfa2746",
    "include/parameters.h": "4e0f016ad52f87b1f1bd72ec2449e43e2d12b8eb098a9facaa4e2a0d56ef5087",
}


def pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def run(upstream_dir, compiler, output=None):
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening

    if not jax.config.x64_enabled:
        raise RuntimeError("Run with JAX_ENABLE_X64=1 for this float64 oracle")
    start = time.perf_counter()
    for path, digest in SOURCES.items():
        actual = hashlib.sha256(
            (upstream_dir / Path(path).name).read_bytes()
        ).hexdigest()
        if actual != digest:
            raise ValueError(f"Upstream SHA256 mismatch: {path}")
    with tempfile.TemporaryDirectory(prefix="gradscf-quack-build-") as directory:
        library = Path(directory) / "oracle.so"
        command = [
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(upstream_dir),
            str(upstream_dir / "RGW_phBSE_static_kernel_A.f90"),
            str(upstream_dir / "phRLR_A.f90"),
            str(upstream_dir / "RGW_phBSE_static_kernel_B.f90"),
            str(upstream_dir / "phRLR_B.f90"),
            "-o",
            str(library),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        lib = ctypes.CDLL(str(library))
        lib.rgw_phbse_static_kernel_a_.restype = None
        lib.phrlr_a_.restype = None
        lib.phrlr_b_.restype = None
        lib.rgw_phbse_static_kernel_b_.restype = None
        rng = np.random.default_rng(83)
        factors = rng.normal(size=(5, 5, 5)) * 0.08
        factors = (factors + factors.transpose(0, 2, 1)) / 2
        screening = np.array([-1.1, -0.6, 0.3, 0.8, 1.2])
        qp = screening + np.array([0.03, 0.04, -0.02, -0.04, -0.05])
        pairs = [(i, a) for i in range(2) for a in range(2, 5)]
        gaps = np.array([screening[a] - screening[i] for i, a in pairs])
        lov = np.stack([factors[:, i, a] for i, a in pairs], axis=1)
        coulomb = lov.T @ lov
        squared, u = np.linalg.eigh(
            np.sqrt(gaps)[:, None]
            * (np.diag(gaps) + 4 * coulomb)
            * np.sqrt(gaps)[None, :]
        )
        omega = np.sqrt(squared)
        xpy = np.sqrt(gaps)[:, None] * u / np.sqrt(omega)[None, :]
        xmy = u * np.sqrt(omega)[None, :] / np.sqrt(gaps)[:, None]
        np.testing.assert_allclose(xpy.T @ xmy, np.eye(6), atol=2e-13, rtol=0)
        rho = np.asfortranarray(np.einsum("Ppq,Pt,ts->pqs", factors, lov, xpy))
        # QuAcK ERI[p,r,q,s] = chemists (pq|rs).
        eri = np.asfortranarray(np.einsum("Ppq,Prs->prqs", factors, factors))
        omega, qp_f = np.asfortranarray(omega), np.asfortranarray(qp)
        ka = np.zeros((6, 6), order="F")
        nbas, nc, no, nv, nr, ns = [ctypes.c_int(i) for i in (5, 0, 2, 3, 0, 6)]
        eta, coupling = ctypes.c_double(0), ctypes.c_double(1)
        dims = [ctypes.byref(i) for i in (nbas, nc, no, nv, nr, ns)]
        lib.rgw_phbse_static_kernel_a_(
            ctypes.byref(eta),
            *dims,
            ctypes.byref(coupling),
            pointer(eri),
            pointer(omega),
            pointer(rho),
            pointer(ka),
        )
        kb = np.zeros((6, 6), order="F")
        lib.rgw_phbse_static_kernel_b_(
            ctypes.byref(eta),
            *dims,
            ctypes.byref(coupling),
            pointer(eri),
            pointer(omega),
            pointer(rho),
            pointer(kb),
        )
        space = bse.make_bse_space(5, 2)
        state = build_static_screening(
            jnp.asarray(screening),
            jnp.asarray(factors),
            occupied=space.occupied,
            virtual=space.virtual,
        )
        errors, matrices = {}, {}
        for singlet in (True, False):
            spin, drpa = ctypes.c_int(1 if singlet else 2), ctypes.c_int(0)
            bare = np.zeros((6, 6), order="F")
            lib.phrlr_a_(
                ctypes.byref(spin),
                ctypes.byref(drpa),
                *dims,
                ctypes.byref(coupling),
                pointer(qp_f),
                pointer(eri),
                pointer(bare),
            )
            expected = bare + ka
            operator = bse.build_tda_operator(
                jnp.asarray(qp),
                jnp.asarray(factors),
                space,
                state,
                singlet=singlet,
                block_size=2,
            )
            actual = np.asarray(operator.apply(jnp.eye(6)))
            label = "singlet" if singlet else "triplet"
            errors[f"{label}_matrix_max_abs_hartree"] = float(
                np.max(np.abs(actual - expected))
            )
            errors[f"{label}_roots_max_abs_hartree"] = float(
                np.max(
                    np.abs(np.linalg.eigvalsh(actual) - np.linalg.eigvalsh(expected))
                )
            )
            errors[f"{label}_screening_correction_max_abs_hartree"] = float(
                np.max(np.abs(actual - bare - ka))
            )
            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
            matrices[f"a_{label}"] = expected
            bare_b = np.zeros((6, 6), order="F")
            lib.phrlr_b_(
                ctypes.byref(spin),
                ctypes.byref(drpa),
                *dims,
                ctypes.byref(coupling),
                pointer(eri),
                pointer(bare_b),
            )
            expected_b = bare_b + kb
            _, coupling_op = bse.build_bse_operators(
                qp, factors, space, state, singlet=singlet, block_size=2
            )
            actual_b = np.asarray(coupling_op.apply(jnp.eye(6)))
            np.testing.assert_allclose(actual_b, expected_b, atol=2e-13, rtol=0)
            matrices[f"b_{label}"] = expected_b
            full_roots = np.sort(
                np.linalg.eigvals(
                    np.block([[expected, expected_b], [-expected_b, -expected]])
                ).real
            )[6:]
            result = bse.run_bse(
                qp,
                screening,
                factors,
                space,
                config=bse.BSEConfig(
                    tda=False, solver="dense", nroots=6, singlet=singlet
                ),
            )
            np.testing.assert_allclose(
                result.excitation_energies, full_roots, atol=2e-12, rtol=0
            )
            errors[f"{label}_coupling_matrix_max_abs_hartree"] = float(
                np.max(np.abs(actual_b - expected_b))
            )
            errors[f"{label}_full_roots_max_abs_hartree"] = float(
                np.max(np.abs(np.asarray(result.excitation_energies) - full_roots))
            )
            matrices[f"full_roots_{label}"] = full_roots
        metadata = {
            "upstream": "https://github.com/pfloos/QuAcK",
            "commit": COMMIT,
            "source_sha256": SOURCES,
            "seed": 83,
            "units": "Hartree",
            "model": "real symmetric synthetic factors; naux=5,nmo=5,nocc=2",
            "screening": "full direct RPA poles, eta=0,lambda=1; distinct screening/QP energies",
            "oracle": "unmodified Fortran phRLR_A/B + RGW_phBSE_static_kernel_A/B; NumPy rho, screening poles and doubled full-BSE eigenvalues",
            "rho_source": "RGW_excitation_density.f90 checked for convention; not compiled",
            "compiler": subprocess.check_output(
                [compiler, "--version"], text=True
            ).splitlines()[0],
            "compiler_flags": "-shared -fPIC -O2 -I <upstream_dir>",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "dtype": "float64",
            "elapsed_seconds": time.perf_counter() - start,
            "command": "PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "
            + shlex.join([sys.executable, *sys.argv]),
            "errors": errors,
        }
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output,
                qp_energy=qp,
                screening_energy=screening,
                factors=factors,
                rpa_omega=omega,
                rho=rho,
                ka=ka,
                kb=kb,
                **matrices,
            )
            output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        print(json.dumps(metadata, indent=2))


def main():
    upstream_dir = Path(tempfile.gettempdir()) / "gradscf-quack-oracle"
    output = (
        Path(__file__).resolve().parents[1] / "bse/data/quack_full_static_seed83.npz"
    )
    upstream_dir.mkdir(parents=True, exist_ok=True)
    for path in SOURCES:
        if not (upstream_dir / Path(path).name).exists():
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "30",
                    f"https://raw.githubusercontent.com/pfloos/QuAcK/{COMMIT}/{path}",
                    "-o",
                    str(upstream_dir / Path(path).name),
                ],
                check=True,
            )
    run(upstream_dir, shutil.which("gfortran") or "gfortran", output)


if __name__ == "__main__":
    main()
