"""Implement miscelaneous utility functions."""

import numpy as np
from typing import Sequence


def print_mean_std(arr: Sequence[float], prefix: str = "") -> None:
    """Print mean and std of an array."""
    print(f"{prefix} {np.mean(arr)=} +/- {np.std(arr)=}")
