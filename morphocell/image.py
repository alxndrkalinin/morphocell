"""Contains a simple class for storing image data."""

try:
    from cupy.cuda.runtime import getDeviceCount

    if getDeviceCount() > 0:
        device_name = "GPU"
        import cupy as xp
    else:
        raise
except Exception:
    device_name = "CPU"
    import numpy as xp


class Image(xp.ndarray):
    """A very simple extension to numpy/cupy ndarray, to contain image data and metadata."""

    def __new__(cls, images, spacing, filename=None):
        """Create new Image object with metadata."""
        obj = xp.asarray(images).view(cls)

        obj.spacing = list(spacing)
        obj.filename = filename
        obj.device = device_name

        return obj

    # https://numpy.org/devdocs/user/basics.subclassing.html#the-role-of-array-finalize
    def __array__finalize__(self, obj):
        """Finalize the subclass of ndarray."""
        self.spacing = getattr(obj, "spacing")
        self.filename = getattr(obj, "filename", None)
