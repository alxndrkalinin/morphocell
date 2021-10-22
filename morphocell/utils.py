"""Implement miscelaneous utility functions."""

import numpy as np
from typing import Sequence
import skimage
from .image import Image


def print_mean_std(arr: Sequence[float], prefix: str = "") -> None:
    """Print mean and std of an array."""
    print(f"{prefix} {np.mean(arr)=} +/- {np.std(arr)=}")


class Device:
    """Provide GPU accelerated libs, if available."""

    def __init__(self):
        """Create image object on specified device, if available."""
        self.gpu = False
        self.xp = np
        self.skimage = skimage
        try:
            import cupy as cp
            import cucim

            if cp.cuda.runtime.getDeviceCount() > 0:
                self.gpu = True
                self.xp = cp
                self.skimage = cucim.skimage
            else:
                raise
        except Exception:
            pass


def cpu_deps():
    """Provide CPU based libraries."""
    return {"xp": np, "xkimage": skimage}


def gpuable(func):
    """Inject device-dependent libraries into function."""

    def wrapper(*args, **kwargs):
        """Wrap function by checking if Image is passed and what device it is on."""
        device = Device()

        if isinstance(args[0], Image) or (device.gpu and isinstance(args[0], device.xp.ndarray)):
            if args[0].device == "GPU" and device.gpu:
                kwargs["deps"] = {
                    "xp": device.xp,
                    "xkimage": device.skimage,
                }
            else:
                kwargs["deps"] = {
                    "xp": np,
                    "xkimage": skimage,
                }

        func(*args, **kwargs)

    return wrapper
