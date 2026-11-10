"""Small bounded synchronous HTTP range reader for remote HDF5 qualification.

This deliberately uses the already-installed ``requests`` backend rather than
fsspec's default async path.  A response is accepted only after its 206 and
Content-Range headers prove that the requested interval was honoured.  The
transfer budget is checked before the response body is iterated, so a server
that ignores Range cannot cause a whole archive to be consumed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any


_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


class RangeProtocolError(RuntimeError):
    """The remote endpoint did not provide the requested bounded range."""


class RangeBudgetExceeded(RuntimeError):
    """The next HTTP response would exceed the declared transfer budget."""


@dataclass
class RangeStats:
    budget_bytes: int
    remote_size: int | None = None
    requests: int = 0
    requested_bytes: int = 0
    transferred_bytes: int = 0
    cache_hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["partial_archive_sha256_verified"] = False
        return result


def _content_range(response: Any) -> tuple[int, int, int | None]:
    value = str(response.headers.get("Content-Range", ""))
    match = _CONTENT_RANGE.match(value.strip())
    if match is None:
        raise RangeProtocolError(f"missing or malformed Content-Range: {value!r}")
    start, end, total = match.groups()
    return int(start), int(end), None if total == "*" else int(total)


class BoundedRangeHTTPFile:
    """Seekable read-only file-like object backed by exact HTTP ranges.

    The reader caches fixed-size blocks.  ``h5py.File`` can therefore perform
    random reads without downloading the complete HDF5 shard.  The budget is
    shared by the initial size probe and every cache fill.
    """

    def __init__(self, url: str, *, block_size: int = 4 * 1024 * 1024,
                 max_bytes: int = 250 * 1024 * 1024, timeout: float = 60,
                 session: Any = None):
        if int(block_size) <= 0 or int(max_bytes) <= 0:
            raise ValueError("block_size and max_bytes must be positive")
        self.url = str(url)
        self.block_size = int(block_size)
        self.stats = RangeStats(budget_bytes=int(max_bytes))
        self._timeout = timeout
        if session is None:
            import requests
            session = requests.Session()
        self._session = session
        self._cache: dict[int, bytes] = {}
        self._position = 0
        self._closed = False
        probe = self._fetch(0, 0)
        del probe
        if self.stats.remote_size is None or self.stats.remote_size <= 0:
            raise RangeProtocolError("range probe did not disclose a positive archive size")

    def _fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError("invalid byte interval")
        expected = end - start + 1
        if self.stats.transferred_bytes + expected > self.stats.budget_bytes:
            raise RangeBudgetExceeded(
                f"range {start}-{end} exceeds transfer budget "
                f"{self.stats.budget_bytes} bytes")
        response = self._session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=self._timeout,
        )
        self.stats.requests += 1
        self.stats.requested_bytes += expected
        try:
            if int(getattr(response, "status_code", 0)) != 206:
                raise RangeProtocolError(
                    f"expected HTTP 206 for bytes {start}-{end}, "
                    f"got {getattr(response, 'status_code', None)}")
            actual_start, actual_end, total = _content_range(response)
            if (actual_start, actual_end) != (start, end):
                raise RangeProtocolError(
                    f"server returned bytes {actual_start}-{actual_end}, "
                    f"requested {start}-{end}")
            if total is not None:
                if self.stats.remote_size is not None and self.stats.remote_size != total:
                    raise RangeProtocolError("remote archive size changed during range reads")
                self.stats.remote_size = total
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != expected:
                raise RangeProtocolError(
                    f"Content-Length {declared} does not match range size {expected}")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=min(1024 * 1024, expected)):
                if chunk:
                    body.extend(chunk)
                    if len(body) > expected:
                        raise RangeProtocolError("response body exceeded Content-Range span")
            if len(body) != expected:
                raise RangeProtocolError(
                    f"response body has {len(body)} bytes, expected {expected}")
            self.stats.transferred_bytes += len(body)
            return bytes(body)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            self.stats.cache_hits += 1
            return self._cache[index]
        start = index * self.block_size
        if start >= self.stats.remote_size:
            return b""
        end = min(self.stats.remote_size, start + self.block_size) - 1
        block = self._fetch(start, end)
        self._cache[index] = block
        return block

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed range file")
        if size is None or int(size) < 0:
            size = self.stats.remote_size - self._position
        size = int(size)
        if size <= 0 or self._position >= self.stats.remote_size:
            return b""
        end_position = min(self.stats.remote_size, self._position + size)
        pieces = []
        while self._position < end_position:
            index = self._position // self.block_size
            block = self._block(index)
            offset = self._position - index * self.block_size
            take = min(end_position - self._position, len(block) - offset)
            if take <= 0:
                raise RangeProtocolError("empty block while reading within remote size")
            pieces.append(block[offset:offset + take])
            self._position += take
        return b"".join(pieces)

    def seek(self, offset: int, whence: int = 0) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed range file")
        if whence == 0:
            position = int(offset)
        elif whence == 1:
            position = self._position + int(offset)
        elif whence == 2:
            position = self.stats.remote_size + int(offset)
        else:
            raise ValueError("invalid whence")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def tell(self) -> int:
        return self._position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        self._closed = True
        close = getattr(self._session, "close", None)
        if close is not None:
            close()

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


__all__ = ["BoundedRangeHTTPFile", "RangeBudgetExceeded", "RangeProtocolError", "RangeStats"]
