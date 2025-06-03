"""Timing utilities for TabPFN."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import numpy as np

times_by_tags: dict[tuple[str, Sequence[str]], list[float]] = defaultdict(list)
_tags: list[str] = []


@contextmanager
def push_tags(*names: str) -> Generator[None, None, None]:
    """Context manager to add tags to the current context.

    Args:
        names: A descriptive name for what is being timed.
    """
    global _tags  # noqa: PLW0603
    old_tags = _tags
    _tags = old_tags + [name for name in names if name not in old_tags]
    try:
        yield
    finally:
        _tags = old_tags


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Context manager to time code execution and print the result.

    Args:
        name: A descriptive name for what is being timed.

    Example:
        ```python
        with timer("model initialization"):
            # some code to time
            initialize_model()
        ```
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        runtime = end_time - start_time
        times_by_tags[(name, tuple(_tags))].append(runtime)


def timing_summary() -> Generator[str, None, None]:
    """Generate a summary of the timing results.

    Yields:
        A string summarizing the timing results for each tag.
    """
    for (name, tags), times in sorted(times_by_tags.items()):
        np_times: np.ndarray = np.array(times)
        (num_times,) = np_times.shape
        total_time = np_times.sum()
        mean_time = np_times.mean()
        std_time = np_times.std()
        min_time = np_times.min()
        max_time = np_times.max()

        yield "-" * 80
        yield f"{name} | {'/'.join(tags)}"
        yield f"    total: {total_time:.4f} s ({num_times} iterations)"
        yield f"  average: {mean_time:.4f} s ± {std_time:.4f} s"
        yield f"  fastest: {min_time:.4f} s"
        yield f"  slowest: {max_time:.4f} s"
        yield ""


__all__ = [
    "push_tags",
    "timer",
    "timing_summary",
]
