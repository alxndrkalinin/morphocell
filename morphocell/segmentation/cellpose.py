"""Implement 3D image segmentation using Cellpose."""

from typing import Optional, List
import numpy.typing as npt

from cellpose import models

from .segment_utils import downsample_and_filter, exclude_touching_objects


def cellpose_eval(
    image: npt.ArrayLike,
    model_type: str = "cyto",
    omni: bool = False,
    channels: Optional[List[int]] = None,
    diameter: Optional[int] = None,
    do_3D: bool = True,
) -> npt.ArrayLike:
    """Run pre-trained Cellpose model and return masks."""
    model = models.Cellpose(gpu=True, model_type="cyto", omni=False)
    masks, _, _, _ = model.eval(image, channels=channels, diameter=diameter, do_3D=True)
    return masks


def cellpose_segment(
    image,
    downscale_factor: float = 0.5,
    model_type: str = "cyto",
    omni: bool = False,
    channels: Optional[List[int]] = None,
    diameter: Optional[int] = None,
    do_3D: bool = True,
    border_value: int = 100,
) -> npt.ArrayLike:
    """Preprocess image, run Cellpose and postprocessing."""
    image = downsample_and_filter(image, downscale_factor=downscale_factor)
    masks = cellpose_eval(image, model_type=model_type, omni=omni, channels=channels, diameter=diameter, do_3D=do_3D)
    masks = exclude_touching_objects(masks, border_value=border_value)
    return masks
