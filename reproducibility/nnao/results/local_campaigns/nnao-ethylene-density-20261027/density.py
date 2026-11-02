"""NumPy evaluation of normalized real spherical GTOs in libcint AO order.

This comparison helper is independent of PySCF. Its s/p/d/f polynomials and
normalization are checked against PySCF on the same frozen basis in run.py.
"""
import math
import numpy as np


def solid_harmonics(l, delta):
    x, y, z = delta.T
    p = math.pi
    if l == 0:
        return np.full((len(x), 1), 1 / np.sqrt(4 * p))
    if l == 1:
        return np.stack((x, y, z), axis=1) * np.sqrt(3 / (4 * p))
    if l == 2:
        return np.stack((np.sqrt(15 / (4 * p)) * x * y,
                         np.sqrt(15 / (4 * p)) * y * z,
                         np.sqrt(5 / (16 * p)) * (2 * z*z - x*x - y*y),
                         np.sqrt(15 / (4 * p)) * x * z,
                         np.sqrt(15 / (16 * p)) * (x*x - y*y)), axis=1)
    if l == 3:
        return np.stack((np.sqrt(35 / (32 * p)) * y * (3 * x*x - y*y),
                         np.sqrt(105 / (4 * p)) * x * y * z,
                         np.sqrt(21 / (32 * p)) * y * (4 * z*z - x*x - y*y),
                         np.sqrt(7 / (16 * p)) * z * (2 * z*z - 3 * x*x - 3 * y*y),
                         np.sqrt(21 / (32 * p)) * x * (4 * z*z - x*x - y*y),
                         np.sqrt(105 / (16 * p)) * z * (x*x - y*y),
                         np.sqrt(35 / (32 * p)) * x * (x*x - 3 * y*y)), axis=1)
    raise NotImplementedError(f"This comparison evaluates s/p/d/f only, got l={l}")


def evaluate_ao(points_bohr, centers_bohr, atom_shells):
    values = []
    for center, shells in zip(centers_bohr, atom_shells):
        delta = points_bohr - center
        r2 = np.sum(delta * delta, axis=1)
        for shell in shells:
            l = int(shell[0])
            rows = np.asarray(shell[1:], dtype=np.float64)
            alpha, c = rows[:, 0], rows[:, 1:]
            power = l + 1.5
            primitive_norm = np.sqrt(2 * (2 * alpha) ** power / math.gamma(power))
            weights = primitive_norm[:, None] * c
            metric = math.gamma(power) / (2 * (alpha[:, None] + alpha[None, :]) ** power)
            weights /= np.sqrt(np.einsum("pi,pq,qi->i", weights, metric, weights))[None, :]
            radial = np.exp(-r2[:, None] * alpha) @ weights
            angular = solid_harmonics(l, delta)
            for column in range(c.shape[1]):
                values.append(radial[:, column, None] * angular)
    return np.concatenate(values, axis=1)


def evaluate_density(points_bohr, centers_bohr, atom_shells, mo, occupations, *, block_size=4096):
    occupied = occupations > 0
    c = mo[:, occupied]
    occupation = occupations[occupied]
    rho = np.empty(len(points_bohr))
    for start in range(0, len(points_bohr), block_size):
        ao = evaluate_ao(points_bohr[start:start+block_size], centers_bohr, atom_shells)
        values = ao @ c
        rho[start:start+block_size] = np.sum(values * values * occupation, axis=1)
    assert np.isfinite(rho).all() and np.all(rho >= 0)
    return rho
