"""Native resources and compatibility imports have one canonical home."""
from pathlib import Path
import tomllib


def test_native_build_and_registration_share_compatibility_objects():
    from gradscf import _native as legacy
    from gradscf._native import build as legacy_build
    from gradscf.integrals import _native
    from gradscf.integrals._native import build
    assert legacy.register_integrals is _native.register_integrals
    assert legacy_build.main is build.main
    assert legacy_build.verify_vendor is build.verify_vendor
    native = Path(build.__file__).resolve().parent
    assert native.parent.name == "integrals"
    assert (native / "CMakeLists.txt").is_file()
    assert (native / "csrc/ffi.cc").is_file()
    build.verify_vendor(native)


def test_native_package_data_covers_all_build_inputs_without_binaries():
    config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["setuptools"]
    patterns = config["package-data"]["gradscf.integrals._native"]
    native = Path("src/gradscf/integrals/_native")
    included = {p for pattern in patterns for p in native.glob(pattern) if p.is_file()}
    required = {native / "CMakeLists.txt", native / "csrc/ffi.cc",
                native / "include/config.h", native / "exports.map", native / "exports.macos"}
    required.update(p for p in (native / "vendor").rglob("*") if p.is_file())
    assert required <= included
    assert not any(p.suffix in {".so", ".dylib", ".o"} for p in included)
