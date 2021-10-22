"""Contains a simple class for storing image data."""

from typing import Optional, Sequence

import numpy as np
from .gpu import get_gpu_info


class Image:
    """A simple class to store NumPy/CuPy image data and metadata."""

    def __init__(self, images, spacing: Sequence, filename: Optional[str] = None, device: str = "CPU"):
        """Create image object with data stored either as NumPy array (CPU) or CuPy array (GPU)."""
        gpu_info = get_gpu_info()

        if device == "GPU" and gpu_info.num_gpus > 0:
            self.data = gpu_info.cp.asarray(images)
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
