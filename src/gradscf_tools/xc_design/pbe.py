"""External closed-shell PBE/PW92 reference for custom_xc acceptance tests.

This is an explicit model definition, not an alternative XC registry. n is in
bohr^-3, sigma=|grad n|^2 in bohr^-8; output is Ha/bohr^3. The declared smooth
vacuum model is n_eff=sqrt(n^2+floor^2), while sigma is unchanged. Published
PBE/PW92 constants are independently checked against the frozen libxc table.
"""
from math import isfinite

import jax.numpy as jnp


def energy_density(n, sigma, *, floor=1e-12):
    if not isfinite(floor) or floor < 0:
        raise ValueError('Declare a finite nonnegative density floor')
    n = jnp.sqrt(jnp.asarray(n)**2+floor**2)
    sigma = jnp.asarray(sigma)
    kf = (3*jnp.pi**2*n)**(1/3)
    s2 = sigma/(4*kf*kf*n*n)
    fx = 1+.804-.804/(1+.2195149727645171*s2/.804)
    exchange = -.75*(3/jnp.pi)**(1/3)*n**(4/3)*fx

    rs = (3/(4*jnp.pi*n))**(1/3)
    a = .0310907
    denominator = 7.5957*jnp.sqrt(rs)+3.5876*rs+1.6382*rs**1.5+.49294*rs**2
    lda = -2*a*(1+.21370*rs)*jnp.log1p(1/(2*a*denominator))
    gamma = .031090690869654895
    ratio = .06672455060314922/gamma
    t2 = sigma/(4*(4*kf/jnp.pi)*n*n)
    coefficient = ratio/jnp.expm1(-lda/gamma)
    at = coefficient*t2
    correction = gamma*jnp.log1p(ratio*t2*(1+at)/(1+at+at*at))
    return exchange+n*(lda+correction)


__all__ = ['energy_density']
