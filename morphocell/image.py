"""Contains a simple class for storing image data."""

import numpy as np

try:
    from cupy.cuda.runtime import getDeviceCount

    if getDeviceCount() > 0:
        import cupy as xp

        device_name = "GPU"
        asnumpy = xp.asnumpy
    else:
        raise
except Exception:
    device_name = "CPU"
    xp = np
    asnumpy = np.asarray


class Image(xp.ndarray):
    """A very simple extension to numpy/cupy ndarray, to contain image data and metadata."""

    def __new__(cls, images, spacing, filename=None, device="CPU"):
        """Create new Image object with metadata."""
        if device == "GPU" and device_name == "GPU":
            obj = xp.asarray(images).view(cls)
        elif device == "CPU" and device_name == "GPU":
            obj = asnumpy(xp.asarray(images).view(cls))
        else:
            obj = np.asarray(images).view(cls)
            device = "CPU"

        obj.spacing = list(spacing)
        obj.filename = filename
        obj.device = device

        return obj

    # https://numpy.org/devdocs/user/basics.subclassing.html#the-role-of-array-finalize
    def __array__finalize__(self, obj):
        """Finalize the subclass of ndarray."""
        self.spacing = getattr(obj, "spacing")
        self.filename = getattr(obj, "filename", None)
        self.device = getattr(obj, "device", "CPU")
