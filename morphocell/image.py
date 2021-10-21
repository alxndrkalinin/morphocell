"""Contains a simple class for storing image data."""

import numpy as np

try:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() > 0:
        device_name = "GPU"
    else:
        raise
except Exception:
    device_name = "CPU"


class Image:
    """A very simple extension to numpy/cupy ndarray, to contain image data and metadata."""

    def __init__(self, images, spacing, filename=None, device="CPU"):
        """Create image object on specified device, if vaailable."""
        if device == "GPU" and device_name == "GPU":
            self.data = cp.asarray(images)
        elif device == "GPU" and device_name == "CPU":
            print("\n GPU is not available! Creating Image on CPU.")
            device = device_name
            self.data = np.asarray(images)
        else:
            self.data = np.asarray(images)

        self.spacing = tuple(spacing)
        self.filename = filename
        self.device = device
