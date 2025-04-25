"""Implements functions to display images and graphs."""

from typing import Tuple
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .image_utils import max_project


def show_image(
    img: npt.ArrayLike,
    figsize: Tuple[int, int] = (10, 10),
    bit_depth: int = 14,
    cmap: str = "gray",
) -> Figure:
    """Display 2D image."""
    plt.figure(figsize=figsize)
    fig = plt.imshow(img, cmap=cmap, vmin=0, vmax=2**bit_depth - 1)
    plt.axis("off")
    return fig


def show_2d(
    img: npt.ArrayLike,
    axis: int = 0,
    figsize: Tuple[int, int] = (10, 10),
    bit_depth: int = 14,
    cmap: str = "gray",
) -> Figure:
    """Max project and display 3D image."""
    return show_image(
        max_project(img, axis), figsize=figsize, bit_depth=bit_depth, cmap=cmap
    )


def show_image_error(
    img: npt.ArrayLike,
    figsize: Tuple[int, int] = (20, 20),
    bit_depth: int = 14,
    cmap: str = "bwr",
) -> Figure:
    """Display error map between teo images."""
    plt.figure(figsize=figsize)
    fig = plt.imshow(
        img, cmap=cmap, vmin=(-(2**bit_depth) - 1), vmax=(2**bit_depth - 1)
    )
    plt.axis("off")
    return fig
