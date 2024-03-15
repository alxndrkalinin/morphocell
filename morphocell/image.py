"""Contains a simple class for storing image data."""
import warnings
from typing import Optional, Sequence

import numpy as np
from .gpu import get_gpu_info, get_device, get_image_method


class Image:
    """A simple class to store NumPy/CuPy image data and metadata."""

    gpu_info = get_gpu_info()

    def __init__(
        self,
        images,
        spacing: Sequence,
        filename: Optional[str] = None,
        device: Optional[str] = None,
        as_float: bool = True,
    ):
        """Create image object with data stored either as NumPy array (CPU) or CuPy array (GPU)."""
        if device is None:
            device = get_device(images)
        else:
            assert device in ["CPU", "GPU"]

        if device.upper() == "GPU" and self.gpu_info["num_gpus"] > 0:
            self.data = self._to_gpu(images)
        elif device.upper() == "GPU" and self.gpu_info["num_gpus"] == 0:
            warnings.warn("GPU requested, but is not available! Creating Image on CPU.")
            self.data = np.asarray(images)
            device = "CPU"
        else:
            self.data = np.asarray(images)

        if as_float:
            img_as_float32 = get_image_method(self.data, "skimage.util.img_as_float32")
            self.data = img_as_float32(self.data)

        self.shape = self.data.shape
        self.spacing = tuple(spacing)
        self.filename = filename
        self.device = device

    def _to_gpu(self, data):
        """Move given array to GPU."""
        data = self.gpu_info["cp"].asarray(data)
        return data

    def to_gpu(self):
        """Move Image data to GPU."""
        if self.gpu_info["num_gpus"] > 0:
            self.data = self._to_gpu(self.data)
            self.device = "GPU"
        else:
            raise ImportError("\n GPU requested, but is not available! Creating Image on CPU.")

    def to_cpu(self):
        """Move (or keep) Image on CPU."""
        try:
            cp = self.gpu_info["cp"]
            if isinstance(self.data, cp.ndarray):
                self.data = cp.asnumpy(self.data)
        except Exception:
            self.data = np.asarray(self.data)
        self.device = "CPU"
