"""Contains a simple class for storing image data."""

from typing import Optional, Sequence

import numpy as np
from .gpu import get_gpu_info, get_device


class Image:
    """A simple class to store NumPy/CuPy image data and metadata."""

    def __init__(self, images, spacing: Sequence, filename: Optional[str] = None, device: Optional[str] = None):
        """Create image object with data stored either as NumPy array (CPU) or CuPy array (GPU)."""
        gpu_info = get_gpu_info()

        if device is None:
            device = get_device(images)
        else:
            assert device in ["CPU", "GPU"]

        if device == "GPU" and gpu_info.num_gpus > 0:
            img_as_float32 = gpu_info.cucim.skimage.util.img_as_float32
            self.data = img_as_float32(gpu_info.cp.asarray(images))
        elif device == "GPU" and gpu_info.num_gpus == 0:
            print("\n GPU requested, but is not available! Creating Image on CPU.")
            self.data = np.asarray(images)
            device = "CPU"
        else:
            self.data = np.asarray(images)

        self.shape = self.data.shape
        self.spacing = tuple(spacing)
        self.filename = filename
        self.device = device
