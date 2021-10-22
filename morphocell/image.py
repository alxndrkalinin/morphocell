"""Contains a simple class for storing image data."""

from typing import Optional, Sequence
import numpy as np
from .utils import Device


class Image:
    """A very simple extension to numpy/cupy ndarray, to contain image data and metadata."""

    def __init__(self, images, spacing: Sequence, filename: Optional[str] = None, device: str = "CPU"):
        """Create image object on specified device, if available."""
        device_config = Device()

        if device == "GPU" and device_config.gpu:
            self.data = device_config.xp.asarray(images)
        elif device == "GPU" and not device_config.gpu:
            print("\n GPU requested, but is not available! Creating Image on CPU.")
            self.data = np.asarray(images)
            device = "CPU"
        else:
            self.data = np.asarray(images)

        self.shape = self.data.shape
        self.spacing = tuple(spacing)
        self.filename = filename
        self.device = device
