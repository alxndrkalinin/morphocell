import numpy as np
from typing import Sequence


def print_mean_std(arr: Sequence[float], prefix: str = "") -> None:
    print(f"{prefix} {np.mean(arr)=} +/- {np.std(arr)=}")
