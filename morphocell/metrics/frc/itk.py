import numpy as np
from scipy.ndimage import rotate

__all__ = [
    "convert_from_numpy",
    "convert_from_itk_image",
    "rotate_image",
]


def convert_from_numpy(array: np.ndarray, spacing=None) -> np.ndarray:
    """Return the input array as an ITK image placeholder."""
    return np.asarray(array)


def convert_from_itk_image(image: np.ndarray) -> np.ndarray:
    """Return numpy array from a pseudo ITK image."""
    return np.asarray(image)


def rotate_image(image: np.ndarray, angle: float, interpolation: str = "nearest") -> np.ndarray:
    """Rotate 3D image around the Z axis by ``angle`` degrees."""
    order = 1 if interpolation == "linear" else 0
    return rotate(image, angle, axes=(1, 2), reshape=False, order=order, mode="constant")
