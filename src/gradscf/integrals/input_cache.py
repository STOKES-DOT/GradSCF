"""Bounded insertion-order cache utility for input assembly."""

from __future__ import annotations

from typing import Any

def _cache_bounded(
    cache: dict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    value: Any,
    maxsize: int,
) -> None:
    if len(cache) >= maxsize:
        cache.pop(next(iter(cache)))
    cache[key] = value


