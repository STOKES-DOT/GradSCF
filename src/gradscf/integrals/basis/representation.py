"""Static Cartesian-to-real-spherical maps in the libcint normalized-sp convention."""
from math import pi, sqrt

import jax.numpy as jnp


def cartesian_to_spherical(topology):
    """Return T such that chi_spherical = chi_cartesian @ T, for s through g shells.

    Shells and contractions retain topology order. s/p Cartesian functions are
    normalized and higher shells have radial normalization. Cartesian d order is
    xx, xy, xz, yy, yz, zz and real spherical d order is xy, yz, z2, xz, x2-y2.
    The AO density embeds as D_cart = T @ D_sph @ T.T.
    """
    if any(l > 4 for l in topology.angular_momenta):
        raise NotImplementedError('The angular transform supports s through g shells (l<=4)')
    ncart = sum((l+1)*(l+2)//2*c for l,c in zip(topology.angular_momenta, topology.contraction_counts))
    nsph = sum((2*l+1)*c for l,c in zip(topology.angular_momenta, topology.contraction_counts))
    result = jnp.zeros((ncart, nsph))
    d = jnp.array([[0.,0.,-sqrt(5/(16*pi)),0.,sqrt(15/(16*pi))],
                   [sqrt(15/(4*pi)),0.,0.,0.,0.],
                   [0.,0.,0.,sqrt(15/(4*pi)),0.],
                   [0.,0.,-sqrt(5/(16*pi)),0.,-sqrt(15/(16*pi))],
                   [0.,sqrt(15/(4*pi)),0.,0.,0.],
                   [0.,0.,sqrt(5/(4*pi)),0.,0.]])
    # Real f harmonics in libcint order m=-3,...,+3. Cartesian rows are
    # xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz.
    a, b = sqrt(35/(32*pi)), sqrt(105/(4*pi))
    c, z, e = sqrt(21/(32*pi)), sqrt(7/(16*pi)), sqrt(105/(16*pi))
    f = jnp.array([[0.,0.,0.,0.,-c,0.,a],
                   [3*a,0.,-c,0.,0.,0.,0.],
                   [0.,0.,0.,-3*z,0.,e,0.],
                   [0.,0.,0.,0.,-c,0.,-3*a],
                   [0.,b,0.,0.,0.,0.,0.],
                   [0.,0.,0.,0.,4*c,0.,0.],
                   [-a,0.,-c,0.,0.,0.,0.],
                   [0.,0.,0.,-3*z,0.,-e,0.],
                   [0.,0.,4*c,0.,0.,0.,0.],
                   [0.,0.,0.,2*z,0.,0.,0.]])
    # Real g harmonics in libcint order m=-4,...,+4. Each row below is one
    # harmonic over Cartesian monomials in descending x and then y powers.
    g0 = 3 / (16 * sqrt(pi))
    g1, g2 = sqrt(45 / (32 * pi)), sqrt(45 / (64 * pi))
    g3, g4 = sqrt(315 / (32 * pi)), sqrt(315 / (256 * pi))
    g = jnp.array([
        [0., 4*g4, 0., 0., 0., 0., -4*g4, 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 3*g3, 0., 0., 0., 0., 0., 0., -g3, 0., 0., 0.],
        [0., -2*g2, 0., 0., 0., 0., -2*g2, 0., 12*g2, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., -3*g1, 0., 0., 0., 0., 0., 0., -3*g1, 0., 4*g1, 0.],
        [3*g0, 0., 0., 6*g0, 0., -24*g0, 0., 0., 0., 0., 3*g0, 0., -24*g0, 0., 8*g0],
        [0., 0., -3*g1, 0., 0., 0., 0., -3*g1, 0., 4*g1, 0., 0., 0., 0., 0.],
        [-g2, 0., 0., 0., 0., 6*g2, 0., 0., 0., 0., g2, 0., -6*g2, 0., 0.],
        [0., 0., g3, 0., 0., 0., 0., -3*g3, 0., 0., 0., 0., 0., 0., 0.],
        [g4, 0., 0., -6*g4, 0., 0., 0., 0., 0., 0., g4, 0., 0., 0., 0.],
    ]).T
    row = column = 0
    for l,count in zip(topology.angular_momenta, topology.contraction_counts):
        block = jnp.eye(2*l+1) if l < 2 else (d, f, g)[l-2]
        for _ in range(count):
            result = result.at[row:row+block.shape[0], column:column+block.shape[1]].set(block)
            row += block.shape[0]
            column += block.shape[1]
    return result


__all__ = ['cartesian_to_spherical']
