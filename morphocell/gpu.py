"""Contains a class for accessing GPU-accelerated libraries."""
from importlib import import_module
import numpy as np


def get_gpu_info():
    """Provide number of available GPUs and GPU libraries (CuPy, CuCIM)."""
    num_gpus = 0
    cp = None
    cucim = None

    try:
        import cupy as cp
        import cucim

        num_gpus = cp.cuda.runtime.getDeviceCount()
    except Exception:
        pass

    return {"num_gpus": num_gpus, "cp": cp, "cucim": cucim}


def asnumpy(array):
    """Move (or keep) array to CPU."""
    try:
        cp = get_gpu_info().cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except Exception:
        return np.asarray(array)


def get_image_method(array, method: str):
    """Return skimage or cucim.skimage method by the argument type/device."""
    module, method = method.rsplit(".", maxsplit=1)

    try:
        if isinstance(array, get_gpu_info().cp.ndarray):
            module = "cucim." + module
    except Exception:
        pass

    return getattr(import_module(module), method)
