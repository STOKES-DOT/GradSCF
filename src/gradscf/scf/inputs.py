"""Compatibility alias for :mod:`gradscf.integrals.assembly`.

Both import paths resolve to the same module object, including monkeypatches.
New production callers should import ``gradscf.integrals.assembly`` directly.
"""

import sys
from ..integrals import assembly as _assembly

sys.modules[__name__] = _assembly
