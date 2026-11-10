import pytest

from gradscf_tools.range_http import (
    BoundedRangeHTTPFile,
    RangeBudgetExceeded,
    RangeProtocolError,
)


class Response:
    def __init__(self, status, start, end, total, payload, *, content_length=True):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
        if content_length:
            self.headers["Content-Length"] = str(len(payload))
        self._payload = payload
        self.closed = False

    def iter_content(self, chunk_size):
        yield self._payload[: max(1, chunk_size // 2)]
        yield self._payload[max(1, chunk_size // 2):]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, data, *, ignore_range=False):
        self.data = data
        self.ignore_range = ignore_range
        self.calls = []

    def get(self, url, *, headers, stream, timeout):
        self.calls.append((url, headers, stream, timeout))
        start, end = (int(value) for value in headers["Range"].removeprefix("bytes=").split("-"))
        payload = self.data[start:end + 1]
        if self.ignore_range:
            response = Response(200, 0, len(self.data) - 1, len(self.data), self.data)
            response.headers.pop("Content-Range")
            return response
        return Response(206, start, end, len(self.data), payload)

    def close(self):
        pass


def test_bounded_reader_requires_exact_206_and_caches_blocks():
    data = bytes(range(32))
    session = Session(data)
    with BoundedRangeHTTPFile("https://example.test/a.h5", block_size=8,
                              max_bytes=32, session=session) as remote:
        assert remote.read(3) == data[:3]
        remote.seek(0)
        assert remote.read(3) == data[:3]
        assert remote.stats.cache_hits == 1
        assert remote.stats.remote_size == len(data)
        assert remote.stats.transferred_bytes == 9  # one-byte probe plus one block


def test_budget_is_checked_before_the_body_is_consumed():
    data = bytes(range(32))
    session = Session(data)
    with pytest.raises(RangeBudgetExceeded):
        with BoundedRangeHTTPFile("https://example.test/a.h5", block_size=8,
                                  max_bytes=5, session=session) as remote:
            remote.read(1)
    assert len(session.calls) == 1  # only the one-byte size probe was sent


def test_ignored_range_is_rejected_before_iterating_a_large_body():
    data = bytes(range(32))
    session = Session(data, ignore_range=True)
    with pytest.raises(RangeProtocolError, match="HTTP 206"):
        BoundedRangeHTTPFile("https://example.test/a.h5", block_size=8,
                             max_bytes=64, session=session)
