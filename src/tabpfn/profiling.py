"""Timing utilities for TabPFN."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import numpy as np

_tags: list[str] = []
_start_times: list[float] = []
_times: dict[str, list[float]] = {}


def start_timer(name: str):
    """Start a timer.

    Args:
        name: A descriptive name for what is being timed.
    """
    assert name not in _tags
    _tags.append(name)
    _start_times.append(time.time())
    path = "/".join(_tags)
    assert len(_tags) == len(_start_times)
    if path not in _times:
        _times[path] = []


def stop_timer(name: str):
    """Stop a timer.

    Args:
        name: A descriptive name for what is being timed.
    """
    path = "/".join(_tags)
    assert _tags.pop() == name
    start_time = _start_times.pop()
    assert len(_tags) == len(_start_times)
    _times[path].append(time.time() - start_time)


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Context manager to time code execution and print the result.

    Args:
        name: A descriptive name for what is being timed.
    """
    start_timer(name)
    try:
        yield
    finally:
        stop_timer(name)


F = TypeVar("F", bound=Callable[..., Any])


def timed(name: str) -> Callable[[F], F]:
    """Decorator to time code execution and print the result.

    Args:
        name: A descriptive name for what is being timed.

    Example:
        ```python
        @timed("model initialization")
        def initialize_model():
            # some code to time
            pass
        ```
    """

    def wrapper(f: F) -> F:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with timer(name):
                return f(*args, **kwargs)

        return cast("F", wrapped)

    return wrapper


def timing_summary() -> Generator[str, None, None]:
    """Generate a summary of the timing results.

    Yields:
        A string summarizing the timing results for each tag.
    """
    for path, times in _times.items():
        assert len(times) > 0

        np_times: np.ndarray = np.array(times)
        (num_times,) = np_times.shape
        mean_time = np_times.mean()
        std_time = np_times.std()
        min_index = np_times.argmin()
        max_index = np_times.argmax()
        min_time = np_times[min_index]
        max_time = np_times[max_index]

        yield "-" * 80
        yield f"{path} ({num_times} {'time' if num_times == 1 else 'iterations'})"
        yield f"  average: {mean_time:.4f} s ± {std_time:.4f} s"
        yield f"  fastest: {min_time:.4f} s on iteration {min_index + 1}"
        yield f"  slowest: {max_time:.4f} s on iteration {max_index + 1}"
        yield ""


__all__ = [
    "start_timer",
    "stop_timer",
    "timed",
    "timer",
    "timing_summary",
]
