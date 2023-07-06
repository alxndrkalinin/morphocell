"""Contains a class for accessing GPU-accelerated libraries."""
from typing import Callable, Dict, Any
from types import ModuleType

import os
import cloudpickle
import multiprocessing as mp

from importlib import import_module

import numpy as np


def get_gpu_info() -> Dict[str, Any]:
    """Provide number of available GPUs and GPU libraries (CuPy, CuCIM)."""
    num_gpus = 0

    try:
        import cupy as cp
        import cucim

        num_gpus = cp.cuda.runtime.getDeviceCount()
    except Exception:
        cp = None
        cucim = None
    return {"num_gpus": num_gpus, "cp": cp, "cucim": cucim}


def get_device(array) -> str:
    """Return current image device."""
    try:
        cp = get_gpu_info()["cp"]
        return "GPU" if isinstance(array, cp.ndarray) else "CPU"
    except Exception:
        return "CPU"


def get_array_module(array) -> ModuleType:
    """Get the NumPy or CuPy method based on argument location."""
    cp = get_gpu_info()["cp"]
    return cp.get_array_module(array) if cp is not None else np


def asnumpy(array):
    """Move (or keep) array to CPU."""
    if isinstance(array, np.ndarray):
        return np.asarray(array)
    try:
        cp = get_gpu_info()["cp"]
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except Exception:
        return np.asarray(array)


def ascupy(array):
    """Move (or keep) array to GPU."""
    try:
        cp = get_gpu_info()["cp"]
        if isinstance(array, np.ndarray):
            return cp.asarray(array)
    except Exception:
        return np.asarray(array)


def get_image_method(array, method: str) -> Callable:
    """Return skimage or cucim.skimage method by the argument type/device."""
    module, method = method.rsplit(".", maxsplit=1)

    try:
        if isinstance(array, get_gpu_info()["cp"].ndarray):
            module = f"cucim.{module}"
    except Exception:
        pass

    return getattr(import_module(module), method)


class RunAsCUDASubprocess:
    """Decorator to run TensorFlow in a separate process."""

    def __init__(self, num_gpus=0, memory_fraction=0.8):
        """Initialize decorator with number of GPUs and amount of available memory."""
        self._num_gpus = num_gpus
        self._memory_fraction = memory_fraction

    @staticmethod
    def _subprocess_code(num_gpus, memory_fraction, fn, args):
        # set the env vars inside the subprocess so that we don't alter the parent env
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see tensorflow issue #152
        try:
            import py3nvml

            num_grabbed = py3nvml.grab_gpus(num_gpus, gpu_fraction=memory_fraction)
        except Exception:
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            num_grabbed = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        assert (
            num_grabbed == num_gpus
        ), f"Could not grab {num_gpus} GPU devices with {memory_fraction * 100}% memory available"

        if os.environ["CUDA_VISIBLE_DEVICES"] == "":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # see tensorflow issues: #16284, #2175

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        """Spawn a separate process to run wrapped TensorFlow function."""

        def wrapped_f(*args):
            with mp.get_context("spawn").Pool(1) as p:
                return p.apply(
                    RunAsCUDASubprocess._subprocess_code,
                    (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args),
                )

        return wrapped_f
