"""Offline source build: ``python -m gradscf.integrals._native.build``."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify_vendor(native):
    """Fail before compilation if any pinned vendor file has changed."""
    manifest = json.loads((native / "vendor" / "manifest.json").read_text())
    for dependency in manifest["dependencies"]:
        for entry in dependency["files"]:
            path = native / "vendor" / dependency["directory"] / entry["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Vendored source checksum mismatch: {path}")


def main():
    import jax

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--blas-library", type=Path, help="Existing LP64 BLAS runtime library (Linux override).")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    native = Path(__file__).resolve().parent
    if not (native / "CMakeLists.txt").is_file():
        parser.error(f"Native sources are missing from {native}")
    verify_vendor(native)
    build_dir = (args.build_dir or native / "build").resolve()
    output_dir = Path(__file__).resolve().parent / "lib"
    blas_args = []
    if args.blas_library is not None:
        if not args.blas_library.is_file():
            parser.error("--blas-library must point to an existing LP64 BLAS library")
        blas_args = [f"-DGRADSCF_BLAS_LIBRARY={args.blas_library.resolve()}"]
    subprocess.run([
        "cmake", "-S", str(native), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", f"-DJAX_INCLUDE_DIR={jax.ffi.include_dir()}",
        f"-DGRADSCF_OUTPUT_DIR={output_dir}", *blas_args,
    ], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)],
                   check=True)


if __name__ == "__main__":
    main()
