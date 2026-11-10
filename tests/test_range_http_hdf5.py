"""Real HDF5 file-object integration over mocked bounded HTTP byte ranges."""
import io

import numpy as np
import pytest

from gradscf_tools.range_http import BoundedRangeHTTPFile
from test_range_http import Session


def test_hdf5_random_reads_preserve_arrays_and_transfer_budget():
    h5py = pytest.importorskip('h5py')
    storage = io.BytesIO()
    expected = np.arange(24, dtype=np.float64).reshape(4,6)
    with h5py.File(storage, 'w') as h:
        h.create_dataset('sample/density', data=expected, compression='gzip')
        h.create_dataset('sample/label', data=np.asarray(-7.5))
    data = storage.getvalue()
    with BoundedRangeHTTPFile('https://example.test/data.h5', block_size=256,
                              max_bytes=len(data)+256, session=Session(data)) as remote:
        with h5py.File(remote, 'r') as h:
            np.testing.assert_array_equal(h['sample/density'][()], expected)
            assert float(h['sample/label'][()]) == -7.5
        assert remote.stats.transferred_bytes <= remote.stats.budget_bytes
        assert remote.stats.cache_hits > 0
