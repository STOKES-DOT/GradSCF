"""Compatibility CLI; prefer python -m gradscf.integrals._native.build."""
from gradscf.integrals._native.build import main, verify_vendor

__all__ = ["main", "verify_vendor"]

if __name__ == "__main__":
    main()
