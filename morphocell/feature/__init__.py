"""Expose feature extraction functions."""

from .voxel_morphometry import regionprops, regionprops_table

__all__ = [
    "regionprops",
    "regionprops_table",
]
