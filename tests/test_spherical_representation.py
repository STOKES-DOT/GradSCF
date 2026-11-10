"""Canonical real spherical embedding independently checks angular integrals."""
import math
from dataclasses import replace

import numpy as np
import pytest

from gradscf.integrals.basis import BasisTopology


def transform(topology):
    from gradscf.integrals.basis import cartesian_to_spherical
    return np.asarray(cartesian_to_spherical(topology))


def test_d_shell_transform_is_orthonormal_in_exact_sphere_metric():
    top = BasisTopology((2,), (1,), (1,), (6,))
    t = transform(top)
    powers = [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
    def moment(a):
        if any(k%2 for k in a): return 0.
        return 2*math.prod(math.gamma((k+1)/2) for k in a)/math.gamma((sum(a)+3)/2)
    metric = np.array([[moment(tuple(x+y for x,y in zip(a,b))) for b in powers] for a in powers])
    np.testing.assert_allclose(t.T@metric@t, np.eye(5), atol=3e-15)
    # Real harmonics have the libcint order xy,yz,z2,xz,x2-y2.
    at_z = np.array([0.,0.,0.,0.,0.,1.])@t
    np.testing.assert_allclose(at_z, [0.,0.,math.sqrt(5/(4*math.pi)),0.,0.], atol=1e-15)


def test_transform_preserves_shell_contraction_order_and_low_l_normalization():
    top = BasisTopology((0,1,2), (3,2,1), (2,1,2), (6,))
    t = transform(top)
    assert t.shape == (17,15)
    np.testing.assert_array_equal(t[:5,:5], np.eye(5))
    d = transform(BasisTopology((2,), (1,), (1,), (6,)))
    np.testing.assert_array_equal(t[5:11,5:10], d)
    np.testing.assert_array_equal(t[11:17,10:15], d)


def test_high_angular_momentum_requires_an_explicit_supported_transform():
    with pytest.raises(NotImplementedError, match='angular|s.*p.*d|l='):
        transform(BasisTopology((3,), (1,), (1,), (6,)))
