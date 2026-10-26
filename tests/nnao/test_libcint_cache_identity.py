"""Fixed primitive cache identity: metadata/file I/O only, no integral kernels."""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import os

import numpy as np
import pytest

from gradscf.integrals import _native
from gradscf.integrals.basis.types import BasisParameters, BasisTopology
from gradscf.model.nnao.integrals.cache import fixed_signature, read_fixed_cache, write_fixed_cache


@pytest.fixture
def native_file(tmp_path, monkeypatch):
    path = tmp_path/"libgradscf_integrals.so"
    path.write_bytes(b"native-library-revision-A")
    handle = SimpleNamespace(_name=str(path))
    monkeypatch.setattr(_native, "register_integrals", lambda: handle)
    return path, handle


def _problem():
    topology = BasisTopology((0, 1), (2, 1), (2, 1), (1, 1), False)
    parameters = BasisParameters(
        (np.asarray([1.2, .3]), np.asarray([.7])),
        (np.eye(2), np.eye(1)),
        np.asarray([[0., 0., 0.], [0., 0., 1.4]]),
        np.asarray([[0., 0., 0.], [0., 0., 1.4]]))
    contracted = replace(topology, contraction_counts=(1, 1))
    model = replace(parameters, coefficients=(np.asarray([[.7], [.2]]), np.ones((1, 1))))
    layout = SimpleNamespace(topology=contracted, parameters=model, theta=np.asarray([.2, -.1]), ecps=())
    coords = np.asarray([[0., 0., 0.], [0., 0., .74]])
    return layout, topology, parameters, coords


def _signature(layout, topology, parameters, coords, **kwargs):
    return fixed_signature(layout, ("H", "H"), coords, "fixed-primitive-fixture",
                           "auxiliary-fixture", topology, parameters, **kwargs)


def test_fixed_signature_tracks_actual_loaded_native_library_bytes(native_file, tmp_path):
    path, handle = native_file
    args = _problem()
    first = _signature(*args)
    stamp = path.stat()
    path.write_bytes(b"native-library-revision-B")  # Same path and size.
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+1_000_000_000))
    second = _signature(*args)
    assert second != first, "Fixed primitive cache identity ignores the loaded native library bytes"
    another = tmp_path/"another-library.so"
    another.write_bytes(b"native-library-revision-C")
    handle._name = str(another)
    assert _signature(*args) != second
    layout, topology, parameters, coords = args
    # An authentic pre-migration signature must not validate a new cache hit.
    legacy = dict(symbols=("H", "H"), coords=coords.tolist(), basis="fixed-primitive-fixture",
        auxbasis="auxiliary-fixture", cart=layout.topology.cart,
        angular_momenta=topology.angular_momenta, primitive_counts=topology.primitive_counts,
        exponents=[a.tolist() for a in parameters.exponents])
    legacy_key = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    n = topology.nao
    archive = write_fixed_cache(tmp_path/"old-cache.npz", legacy_key,
        np.eye(n), np.eye(n), np.ones((1, n*(n+1)//2)))
    with pytest.raises(ValueError, match="(?i)rebuild|regenerat|recomput|重建"):
        read_fixed_cache(archive, _signature(*args), "host")


def test_fixed_signature_tracks_primitive_centers_independently_of_nuclei(native_file):
    layout, topology, parameters, coords = _problem()
    first = _signature(layout, topology, parameters, coords)
    centers = parameters.centers.copy()
    centers[0, 0] += .01  # Bohr; external/nuclear R and exponents are unchanged.
    moved = replace(parameters, centers=centers)
    np.testing.assert_array_equal(moved.nuclear_coords, parameters.nuclear_coords)
    for left, right in zip(moved.exponents, parameters.exponents):
        np.testing.assert_array_equal(left, right)
    assert _signature(layout, topology, moved, coords) != first, (
        "Fixed primitive cache identity ignores independent physical primitive centers")
    nuclei = parameters.nuclear_coords.copy()
    nuclei[1, 2] += .02
    assert _signature(layout, topology, replace(parameters, nuclear_coords=nuclei), coords) != first
    assert _signature(layout, replace(topology, nuclear_charges=(2, 1)), parameters, coords) != first


def test_fixed_signature_is_independent_of_trainable_contractions_and_theta(native_file):
    layout, topology, parameters, coords = _problem()
    before = _signature(layout, topology, parameters, coords)
    trained = SimpleNamespace(**{**vars(layout), "theta": layout.theta+np.asarray([.1, -.2]),
        "parameters": replace(layout.parameters,
            coefficients=(layout.parameters.coefficients[0]+.13, layout.parameters.coefficients[1]-.07))})
    # The actual primitive parameters remain identity contractions. Model
    # outputs are applied later, and do not alter these fixed S/h/RI factors.
    assert _signature(trained, topology, parameters, coords) == before


def test_fixed_signature_binds_auxiliary_parameters_and_normalization_source(native_file, tmp_path, monkeypatch):
    from gradscf.integrals.basis import normalization
    from gradscf.model.nnao.integrals import context

    layout, topology, parameters, coords = args = _problem()
    auxiliary_topology = BasisTopology((0,), (2,), (1,), (1, 1), False)
    auxiliary_parameters = BasisParameters((np.asarray([2.1, .45]),),
        (np.asarray([[.6], [.4]]),), np.asarray([[.2, -.1, .3]]), parameters.nuclear_coords.copy())
    source = tmp_path/"normalization.py"
    source.write_text("# normalization identity revision A\n")
    monkeypatch.setattr(normalization, "__file__", str(source))

    def signature(at=auxiliary_topology, ap=auxiliary_parameters):
        try:
            return _signature(*args, auxiliary_topology=at, auxiliary_parameters=ap)
        except TypeError as error:
            pytest.fail(f"fixed_signature must accept actual prepared auxiliary metadata: {error}")

    original = signature()
    variants = [replace(auxiliary_parameters, exponents=(auxiliary_parameters.exponents[0]*1.05,)),
        replace(auxiliary_parameters, coefficients=(np.asarray([[.65], [.4]]),)),
        replace(auxiliary_parameters, centers=auxiliary_parameters.centers+np.asarray([[.01, 0., 0.]]))]
    for changed in variants:
        assert signature(ap=changed) != original
    assert signature(at=replace(auxiliary_topology, angular_momenta=(1,))) != original
    stamp = source.stat()
    source.write_text("# normalization identity revision B\n")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns+1_000_000_000))
    assert signature() != original

    # Metadata sentinels stop before storage/numerical work. No value kernel
    # is replaced with a fake result; only an auxiliary metadata provider is used.
    prepared = []
    def prepare_auxiliary(atom, basis, **kwargs):
        prepared.append(basis)
        return auxiliary_topology, auxiliary_parameters
    monkeypatch.setattr(context.integrals, "prepare_basis", prepare_auxiliary)
    monkeypatch.setattr(context, "primitive_basis", lambda *unused: (topology, parameters))
    expected = signature()
    class CacheBoundary(Exception):
        pass
    def at_cache(path, current, storage):
        assert prepared == ["auxiliary-fixture"]
        assert current == expected
        raise CacheBoundary
    monkeypatch.setattr(context, "read_fixed_cache", at_cache)
    with pytest.raises(CacheBoundary):
        context.prepare_fixed_context(layout, ("H", "H"), coords,
            basis_family="fixed-primitive-fixture", jk_backend="df", auxbasis="auxiliary-fixture",
            integral_cache=tmp_path/"metadata-only.npz", cache_storage="host")
    def forbidden_auxiliary(*unused, **kwargs):
        pytest.fail("A direct context must not parse an unused auxiliary basis")
    def at_plan(*unused, **kwargs):
        raise CacheBoundary
    monkeypatch.setattr(context.integrals, "prepare_basis", forbidden_auxiliary)
    monkeypatch.setattr(context.integrals, "make_plan", at_plan)
    with pytest.raises(CacheBoundary):
        context.prepare_fixed_context(layout, ("H", "H"), coords,
            basis_family="fixed-primitive-fixture", jk_backend="direct", auxbasis="unused-auxiliary")
