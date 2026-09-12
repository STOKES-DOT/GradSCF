"""Private native library loading; public access is through integrals backends."""

from functools import lru_cache
import ctypes
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def register_integrals():
    """Load only GradSCF's own CPU library and keep its handle alive."""
    import jax

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = Path(__file__).parent / "lib" / ("libgradscf_integrals" + suffix)
    if not library.is_file():
        raise RuntimeError(
            "GradSCF native CPU integrals are not built. Run "
            "`python -m gradscf.integrals._native.build` (requires CMake and a C/C++ compiler)."
        )
    handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
    jax.ffi.register_ffi_target(
        "gradscf_integrals_cpu_v1", jax.ffi.pycapsule(handle.GradSCFIntegrals),
        platform="cpu", api_version=1,
    )
    return handle
