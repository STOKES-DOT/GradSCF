"""Independent algebra checks for the manuscript's Theory.

The dimensionless three-primitive/two-AO model below is not a molecular
calculation, GradSCF regression, or performance benchmark. NumPy float64 is
used to compare separately evaluated tensor contractions and reconverged
energy finite differences with the stated stationary and adjoint formulas.
Joint projection checks use linear primitive-tensor perturbations, a fixed
auxiliary whitening operator, and prescribed density variations.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    alphas = np.array([0.2, 0.8, 3.0])
    s0 = (2 * np.sqrt(alphas[:, None] * alphas[None, :])
          / (alphas[:, None] + alphas[None, :])) ** 1.5
    eig, vec = np.linalg.eigh(s0)
    sqrt_s0 = (vec * np.sqrt(eig)) @ vec.T
    h0 = sqrt_s0 @ np.diag([-1.3, -0.5, 0.4]) @ sqrt_s0
    bp = rng.normal(size=(4, 3, 3))
    bp = 0.025 * (bp + bp.transpose(0, 2, 1)) / 2
    core = np.array([1.0, 0.2, 0.1])
    ref = np.array([0.1, 0.3, 1.0])
    direction = np.array([0.2, -0.1, 0.05])
    errors = {}

    def normalized(u):
        d = u / np.linalg.norm(u)
        norm = np.sqrt(d @ s0 @ d)
        jac_d = (np.eye(3) - np.outer(d, d)) / np.linalg.norm(u)
        jac_bar = np.eye(3) / norm - np.outer(d, s0 @ d) / norm**3
        return d / norm, jac_bar @ jac_d

    def transform(eta):
        active, jac = normalized(ref + eta * direction)
        fixed, _ = normalized(core)
        t = np.column_stack((fixed, active))
        dt = np.column_stack((np.zeros(3), jac @ direction))
        return t, dt

    def data(eta):
        t, dt = transform(eta)
        s, h = t.T @ s0 @ t, t.T @ h0 @ t
        b = np.einsum("ai,Qab,bj->Qij", t, bp, t)
        ds = dt.T @ s0 @ t + t.T @ s0 @ dt
        dh = dt.T @ h0 @ t + t.T @ h0 @ dt
        db = (np.einsum("ai,Qab,bj->Qij", dt, bp, t)
              + np.einsum("ai,Qab,bj->Qij", t, bp, dt))
        return t, s, h, b, ds, dh, db

    def jk(b, p):
        rho = np.einsum("Qij,ji->Q", b, p)
        j = np.einsum("Q,Qij->ij", rho, b)
        k = np.einsum("Qik,kl,Qjl->ij", b, p, b)
        return (j + j.T) / 2, (k + k.T) / 2

    def orbitals(p, eta):
        _, s, h, b, _, _, _ = data(eta)
        j, k = jk(b, p)
        f = h + j - 0.5 * k
        val, mat = np.linalg.eigh(s)
        x = (mat / np.sqrt(val)) @ mat.T
        eps, u = np.linalg.eigh(x.T @ f @ x)
        return eps, x @ u, f

    def phi(p, eta):
        _, c, _ = orbitals(p, eta)
        return 2 * c[:, :1] @ c[:, :1].T

    def energy(p, eta):
        _, _, h, b, _, _, _ = data(eta)
        j, k = jk(b, p)
        return np.trace(p @ h) + 0.5 * np.trace(p @ j) - 0.25 * np.trace(p @ k)

    def solve(eta):
        p = np.zeros((2, 2))
        for _ in range(200):
            new = phi(p, eta)
            if np.linalg.norm(new - p) < 2e-14:
                return new
            p = 0.7 * new + 0.3 * p
        raise RuntimeError("Synthetic SCF failed to converge")

    step = 1e-5
    eta = 0.07
    t, s, h, b, ds, dh, db = data(eta)
    p = solve(eta)
    eps, c, f = orbitals(p, eta)
    errors["raw_fixed_point_residual"] = float(np.linalg.norm(p - phi(p, eta)))
    errors["physical_normalization"] = float(np.max(np.abs(np.diag(s) - 1)))
    u = ref + eta * direction
    _, jac = normalized(u)
    fd_jac = np.column_stack([
        (normalized(u + step * np.eye(3)[j])[0]
         - normalized(u - step * np.eye(3)[j])[0]) / (2 * step)
        for j in range(3)
    ])
    errors["normalization_jacobian"] = float(np.max(np.abs(jac - fd_jac)))
    errors["df_factor_derivative"] = float(np.max(np.abs(
        db - (data(eta + step)[3] - data(eta - step)[3]) / (2 * step))))
    errors["overlap_derivative"] = float(np.max(np.abs(
        ds - (data(eta + step)[1] - data(eta - step)[1]) / (2 * step))))
    errors["one_electron_derivative"] = float(np.max(np.abs(
        dh - (data(eta + step)[2] - data(eta - step)[2]) / (2 * step))))

    i0 = np.einsum("Qab,Qcd->abcd", bp, bp)
    eri = np.einsum("ai,bj,ck,dl,abcd->ijkl", t, t, t, t, i0)
    pair_t = np.kron(t, t)
    pair_eri = pair_t.T @ i0.reshape(9, 9) @ pair_t
    errors["ordered_pair_eri_projection"] = float(np.max(np.abs(
        eri - pair_eri.reshape(2, 2, 2, 2))))
    _, dt = transform(eta)
    d_eri = (np.einsum("ai,bj,ck,dl,abcd->ijkl", dt, t, t, t, i0)
             + np.einsum("ai,bj,ck,dl,abcd->ijkl", t, dt, t, t, i0)
             + np.einsum("ai,bj,ck,dl,abcd->ijkl", t, t, dt, t, i0)
             + np.einsum("ai,bj,ck,dl,abcd->ijkl", t, t, t, dt, i0))

    def projected_eri(eta_local):
        z, _ = transform(eta_local)
        return np.einsum("ai,bj,ck,dl,abcd->ijkl", z, z, z, z, i0)

    errors["four_factor_eri_derivative"] = float(np.max(np.abs(
        d_eri - (projected_eri(eta + step) - projected_eri(eta - step))
        / (2 * step))))
    j4, k4 = np.einsum("ijkl,kl->ij", eri, p), np.einsum("ikjl,kl->ij", eri, p)
    jdf, kdf = jk(b, p)
    dp = t @ p @ t.T
    jp = np.einsum("abcd,cd->ab", i0, dp)
    kp = np.einsum("acbd,cd->ab", i0, dp)
    errors["df_j"] = float(np.max(np.abs(j4 - jdf)))
    errors["df_k"] = float(np.max(np.abs(k4 - kdf)))
    errors["direct_projection_j"] = float(np.max(np.abs(j4 - t.T @ jp @ t)))
    errors["direct_projection_k"] = float(np.max(np.abs(k4 - t.T @ kp @ t)))

    def primitive_jk(density):
        return (np.einsum("abcd,cd->ab", i0, density),
                np.einsum("acbd,cd->ab", i0, density))

    def projected_jk(z, density):
        vj, vk = primitive_jk(z @ density @ z.T)
        return z.T @ vj @ z, z.T @ vk @ z

    p_plus, p_minus = solve(eta + step), solve(eta - step)
    dp_response = (p_plus - p_minus) / (2 * step)
    t_plus, _ = transform(eta + step)
    t_minus, _ = transform(eta - step)
    for kind, response in [("fixed_density", np.zeros_like(p)),
                           ("total_response", dp_response)]:
        delta_dp = dt @ p @ t.T + t @ response @ t.T + t @ p @ dt.T
        delta_jp, delta_kp = primitive_jk(delta_dp)
        delta_j = dt.T @ jp @ t + t.T @ delta_jp @ t + t.T @ jp @ dt
        delta_k = dt.T @ kp @ t + t.T @ delta_kp @ t + t.T @ kp @ dt
        density_plus = p if kind == "fixed_density" else p_plus
        density_minus = p if kind == "fixed_density" else p_minus
        plus_j, plus_k = projected_jk(t_plus, density_plus)
        minus_j, minus_k = projected_jk(t_minus, density_minus)
        errors[f"projected_j_{kind}_derivative"] = float(np.max(np.abs(
            delta_j - (plus_j - minus_j) / (2 * step))))
        errors[f"projected_k_{kind}_derivative"] = float(np.max(np.abs(
            delta_k - (plus_k - minus_k) / (2 * step))))

    # Two different linear maps act on orbital-pair and auxiliary indices.
    raw_three = rng.normal(size=(5, 3, 3))
    raw_three = (raw_three + raw_three.transpose(0, 2, 1)) / 2
    aux = rng.normal(size=(5, 5))
    metric = aux @ aux.T + np.eye(5)
    whitening = np.linalg.inv(np.linalg.cholesky(metric))
    whitened_primitive = np.einsum("QR,Rab->Qab", whitening, raw_three)
    whiten_first = np.einsum("ai,Qab,bj->Qij", t, whitened_primitive, t)
    contract_first = np.einsum("QR,ai,Rab,bj->Qij",
                               whitening, t, raw_three, t)
    errors["df_whitening_contraction_commutation"] = float(np.max(np.abs(
        whiten_first - contract_first)))

    rho = np.einsum("Qij,ji->Q", b, p)
    drho = np.einsum("Qij,ji->Q", db, p)
    dj = np.einsum("Q,Qij->ij", rho, db) + np.einsum("Q,Qij->ij", drho, b)
    dk = (np.einsum("Qik,kl,Qjl->ij", db, p, b)
          + np.einsum("Qik,kl,Qjl->ij", b, p, db))
    explicit = np.trace(p @ dh) + 0.5 * np.trace(p @ dj) - 0.25 * np.trace(p @ dk)
    w = 2 * (c[:, :1] * eps[:1]) @ c[:, :1].T
    stationary = explicit - np.trace(w @ ds)
    fd_energy = (energy(solve(eta + step), eta + step)
                 - energy(solve(eta - step), eta - step)) / (2 * step)
    errors["stationary_energy_vs_reconverged_fd"] = float(abs(stationary - fd_energy))

    def residual(p_local, eta_local):
        return (p_local - phi(p_local, eta_local)).reshape(-1)

    eye = np.eye(4)
    a = np.column_stack([
        (residual(p + step * v.reshape(2, 2), eta)
         - residual(p - step * v.reshape(2, 2), eta)) / (2 * step)
        for v in eye
    ])
    g = (residual(p, eta + step) - residual(p, eta - step)) / (2 * step)
    lam = np.linalg.solve(a.T, f.reshape(-1))
    adjoint = explicit - g @ lam
    errors["adjoint_vs_reconverged_fd"] = float(abs(adjoint - fd_energy))
    errors["adjoint_vs_stationary"] = float(abs(adjoint - stationary))

    # Exact common log scaling leaves a normalized same-shell overlap unchanged.
    overlap_errors, overlap_derivatives, gaussian_derivatives = [], [], []
    radii = np.array([0.2, 0.8, 1.5])
    for ell in range(3):
        def shell_overlap(beta):
            alpha = alphas * np.exp(beta)
            return (2 * np.sqrt(alpha[:, None] * alpha[None, :])
                    / (alpha[:, None] + alpha[None, :])) ** (ell + 1.5)

        overlap_errors.append(np.max(np.abs(shell_overlap(0.19) - shell_overlap(0.))))
        overlap_derivatives.append(np.max(np.abs(
            (shell_overlap(step) - shell_overlap(-step)) / (2 * step))))
        # Gamma(ell+3/2) in closed form; the radial factor has unit r^2 dr norm.
        gamma = np.sqrt(np.pi) * np.prod(np.arange(ell + 1) + 0.5)

        def gaussian(beta):
            alpha = 0.7 * np.exp(beta)
            norm = np.sqrt(2 * (2 * alpha) ** (ell + 1.5) / gamma)
            return norm * radii**ell * np.exp(-alpha * radii**2)

        analytic = ((ell + 1.5) / 2 - 0.7 * radii**2) * gaussian(0.)
        gaussian_derivatives.append(np.max(np.abs(
            analytic - (gaussian(step) - gaussian(-step)) / (2 * step))))
    errors["common_shell_log_scale_overlap_invariance"] = float(max(overlap_errors))
    errors["common_shell_log_scale_overlap_derivative"] = float(max(overlap_derivatives))
    errors["normalized_gaussian_log_scale_derivative"] = float(max(gaussian_derivatives))

    # The following beta path is algebraic, not molecular integral evaluation.
    delta_s0 = np.array([[0., .03, -.02], [.03, 0., .01], [-.02, .01, 0.]])
    delta_h0 = rng.normal(size=(3, 3))
    delta_h0 = .02 * (delta_h0 + delta_h0.T) / 2
    delta_i0 = rng.normal(size=(3, 3, 3, 3))
    delta_i0 = (delta_i0 + delta_i0.swapaxes(0, 1)) / 2
    delta_i0 = (delta_i0 + delta_i0.swapaxes(2, 3)) / 2
    delta_i0 = .001 * (delta_i0 + delta_i0.transpose(2, 3, 0, 1)) / 2
    delta_three = rng.normal(size=raw_three.shape)
    delta_three = .03 * (delta_three + delta_three.transpose(0, 2, 1)) / 2
    raw_columns = np.column_stack((core, ref + eta * direction))
    delta_columns = np.column_stack((np.zeros(3), direction))

    def joint_data(beta):
        metric_local = s0 + beta * delta_s0
        columns = raw_columns + beta * delta_columns
        norms = np.sqrt(np.einsum("ai,ab,bi->i", columns, metric_local, columns))
        return (columns / norms, metric_local, h0 + beta * delta_h0,
                i0 + beta * delta_i0, raw_three + beta * delta_three)

    def joint_fd(function):
        return (function(step) - function(-step)) / (2 * step)

    tj, sj, hj, ij, three_j = joint_data(0.)
    norms = np.sqrt(np.einsum("ai,ab,bi->i", raw_columns, sj, raw_columns))
    delta_norm2 = (2 * np.sum(delta_columns * (sj @ raw_columns), axis=0)
                   + np.einsum("ai,ab,bi->i", raw_columns, delta_s0, raw_columns))
    dtj = delta_columns / norms - raw_columns * delta_norm2 / (2 * norms**3)
    errors["joint_normalization_derivative"] = float(np.max(np.abs(
        dtj - joint_fd(lambda beta: joint_data(beta)[0]))))
    for name, index, primitive, delta in (
            ("overlap", 1, sj, delta_s0), ("one_electron", 2, hj, delta_h0)):
        analytic = dtj.T @ primitive @ tj + tj.T @ delta @ tj + tj.T @ primitive @ dtj
        def projected(beta):
            values = joint_data(beta)
            return values[0].T @ values[index] @ values[0]
        errors[f"joint_{name}_derivative"] = float(np.max(np.abs(analytic - joint_fd(projected))))

    def four_index(z, primitive):
        return np.einsum("ai,bj,ck,dl,abcd->ijkl", z, z, z, z, primitive)

    delta_eri = four_index(tj, delta_i0)
    for leg in range(4):
        factors = [tj] * 4
        factors[leg] = dtj
        delta_eri += np.einsum("ai,bj,ck,dl,abcd->ijkl", *factors, ij)
    errors["joint_four_index_derivative"] = float(np.max(np.abs(
        delta_eri - joint_fd(lambda beta: four_index(joint_data(beta)[0], joint_data(beta)[3])))))
    pair_j, delta_pair_j = np.kron(tj, tj), np.kron(dtj, tj) + np.kron(tj, dtj)
    delta_pair_eri = (delta_pair_j.T @ ij.reshape(9, 9) @ pair_j
                      + pair_j.T @ delta_i0.reshape(9, 9) @ pair_j
                      + pair_j.T @ ij.reshape(9, 9) @ delta_pair_j)
    errors["joint_ordered_pair_variation"] = float(np.max(np.abs(
        delta_pair_eri.reshape(2, 2, 2, 2) - delta_eri)))

    bj = np.einsum("QR,Rab->Qab", whitening, three_j)
    delta_bj = np.einsum("QR,Rab->Qab", whitening, delta_three)
    factor_j = np.einsum("ai,Qab,bj->Qij", tj, bj, tj)
    delta_factor_j = (np.einsum("ai,Qab,bj->Qij", dtj, bj, tj)
                      + np.einsum("ai,Qab,bj->Qij", tj, delta_bj, tj)
                      + np.einsum("ai,Qab,bj->Qij", tj, bj, dtj))

    def joint_factors(beta):
        z, _, _, _, three = joint_data(beta)
        return np.einsum("QR,ai,Rab,bj->Qij", whitening, z, three, z)

    errors["joint_df_factor_derivative"] = float(np.max(np.abs(
        delta_factor_j - joint_fd(joint_factors))))
    errors["joint_df_whitening_contraction_commutation"] = float(np.max(np.abs(
        factor_j - joint_factors(0.))))

    def tensor_jk(primitive, density):
        return (np.einsum("abcd,cd->ab", primitive, density),
                np.einsum("acbd,cd->ab", primitive, density))

    density_j = tj @ p @ tj.T
    primitive_j, primitive_k = tensor_jk(ij, density_j)
    for kind, response in (("fixed_density", np.zeros_like(p)),
                           ("prescribed_density", np.array([[.06, -.02], [-.02, .03]]))):
        delta_density_j = dtj @ p @ tj.T + tj @ response @ tj.T + tj @ p @ dtj.T
        djp, dkp = tensor_jk(ij, delta_density_j)
        intrinsic_j, intrinsic_k = tensor_jk(delta_i0, density_j)
        direct_derivatives = (dtj.T @ primitive_j @ tj + tj.T @ (djp + intrinsic_j) @ tj
                              + tj.T @ primitive_j @ dtj,
                              dtj.T @ primitive_k @ tj + tj.T @ (dkp + intrinsic_k) @ tj
                              + tj.T @ primitive_k @ dtj)
        rho_j = np.einsum("Qij,ji->Q", factor_j, p)
        delta_rho_j = (np.einsum("Qij,ji->Q", delta_factor_j, p)
                       + np.einsum("Qij,ji->Q", factor_j, response))
        df_derivatives = (np.einsum("Q,Qij->ij", rho_j, delta_factor_j)
                          + np.einsum("Q,Qij->ij", delta_rho_j, factor_j),
                          np.einsum("Qik,kl,Qjl->ij", delta_factor_j, p, factor_j)
                          + np.einsum("Qik,kl,Qjl->ij", factor_j, response, factor_j)
                          + np.einsum("Qik,kl,Qjl->ij", factor_j, p, delta_factor_j))
        for mode, derivatives in (("projected", direct_derivatives), ("df", df_derivatives)):
            def evaluated(beta):
                z, _, _, primitive, _ = joint_data(beta)
                density = p + beta * response
                if mode == "df":
                    return np.array(jk(joint_factors(beta), density))
                contracted = four_index(z, primitive)
                return np.array((np.einsum("ijkl,kl->ij", contracted, density),
                                 np.einsum("ikjl,kl->ij", contracted, density)))
            expected = joint_fd(evaluated)
            for leg, label in enumerate(("j", "k")):
                errors[f"joint_{mode}_{label}_{kind}_derivative"] = float(np.max(np.abs(
                    derivatives[leg] - expected[leg])))
    limits = {name: (1e-8 if "energy" in name or "adjoint" in name
                     else 1e-9 if "derivative" in name or "jacobian" in name
                     else 1e-12) for name in errors}
    passed = all(errors[name] <= limits[name] for name in errors)
    result = {
        "scope": "dimensionless algebraic model; not a molecular or GradSCF run",
        "backend": "NumPy on CPU",
        "architecture": platform.machine(),
        "dtype": "float64",
        "seed": 0,
        "model": {"primitive_count": 3, "ao_count": 2, "df_rank": 4, "electron_count": 2},
        "joint_variation_scope": "linear primitive tensors, fixed whitening, prescribed density variation; not a joint SCF test",
        "joint_df_rank": int(whitening.shape[0]),
        "finite_difference_step": step,
        "elapsed_seconds": time.perf_counter() - started,
        "errors": errors,
        "absolute_tolerances": limits,
        "passed": passed,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not passed:
        raise AssertionError("Theory identity checks failed")


if __name__ == "__main__":
    main()
