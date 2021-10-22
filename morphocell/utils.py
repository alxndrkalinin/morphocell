"""Implement miscelaneous utility functions."""

from typing import Sequence

import numpy as np


def print_mean_std(arr: Sequence[float], prefix: str = "") -> None:
    """Print mean and std of an array."""
    print(f"{prefix} {np.mean(arr)=} +/- {np.std(arr)=}")
