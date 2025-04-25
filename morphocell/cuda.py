"""Contains a class for accessing CUDA-accelerated libraries."""

from types import ModuleType

import warnings
import os

import numpy as np


class CUDAManager:
    """Manages CUDA resources."""

    _instance = None

    def __new__(cls):
        """Ensure only one instance of CUDAManager is created."""
        if cls._instance is None:
            cls._instance = super(CUDAManager, cls).__new__(cls)
            cls._instance.init_gpu()
        return cls._instance

    def init_gpu(self):
        """Initialize GPU resources."""
        try:
            import cupy as cp
            import cucim

            self.cp = cp
            self.cucim = cucim
            self.num_gpus = cp.cuda.runtime.getDeviceCount()
        except ImportError:
            self.cp = self.cucim = None
            self.num_gpus = 0
            warnings.warn("CuPy or CuCIM is not installed. Falling back to CPU.")
        except Exception:
            self.cp = self.cucim = None
            self.num_gpus = 0
            warnings.warn(
                "Unable to detect CUDA-compatible GPU at the runtime. Check that driver is installed and GPU is visible. Falling back to CPU."
            )

    def get_cp(self):
        """Return CuPy if available."""
        return self.cp

    def get_num_gpus(self):
        """Return number of available GPUs."""
        return self.num_gpus


def get_device(array) -> str:
    """Return current image device."""
    cp = CUDAManager().get_cp()
    if cp is not None and hasattr(array, "device"):
        return "GPU"
    return "CPU"


def to_device(array, device):
    """Move array to the requested device."""
    cp = CUDAManager().get_cp()
    if device == "GPU":
        if cp is not None:
            return cp.asarray(array)
        else:
            raise RuntimeError("GPU requested but not available.")
    elif device == "CPU":
        return np.asarray(array)
    else:
        raise ValueError(
            f"Device should be 'CPU' or 'GPU', unknown requested: {device}."
        )


def to_same_device(source_array, reference_array):
    """Move the source_array to the same device as reference_array."""
    target_device = get_device(reference_array)
    return to_device(source_array, target_device)


def check_same_device(*arrays):
    """Check all provided arrays are on the same device."""
    devices = [get_device(a) for a in arrays]
    unique_devices = sorted(set(devices))
    if len(unique_devices) != 1:
        raise ValueError(
            f"All inputs must be on the same device, but found: {unique_devices}"
        )


def get_array_module(array) -> ModuleType:
    """Get the NumPy or CuPy method based on argument location."""
    cp = CUDAManager().get_cp()
    if cp is not None:
        return cp.get_array_module(array)
    return np


def asnumpy(array):
    """Move (or keep) array to CPU."""
    cp = CUDAManager().get_cp()
    if isinstance(array, np.ndarray):
        return np.asarray(array)
    elif cp is not None and hasattr(array, "device"):
        return cp.asnumpy(array)
    return np.asarray(array)


def ascupy(array):
    """Move (or keep) array to GPU."""
    cp = CUDAManager().get_cp()
    if cp is not None:
        return cp.asarray(array)
    raise RuntimeError("GPU requested but not available.")


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
        except ImportError:
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            num_grabbed = 0
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        assert num_grabbed == num_gpus, (
            f"Could not grab {num_gpus} GPU devices with {memory_fraction * 100}% memory available"
        )

        if os.environ["CUDA_VISIBLE_DEVICES"] == "":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                "-1"  # see tensorflow issues: #16284, #2175
            )

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        import cloudpickle

        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        """Spawn a separate process to run wrapped TensorFlow function."""

        def wrapped_f(*args):
            """Wrap the function to run in a separate process."""
            import cloudpickle
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(1) as p:
                return p.apply(
                    RunAsCUDASubprocess._subprocess_code,
                    (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args),
                )

        return wrapped_f
