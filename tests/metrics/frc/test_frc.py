from __future__ import annotations

import numpy as np
import pytest

from morphocell.metrics.frc import (
    calculate_frc,
    calculate_fsc,
    five_crop_resolution,
    grid_crop_resolution,
    frc_resolution_difference,
)
from morphocell.cuda import ascupy, CUDAManager
from morphocell.skimage import data


@pytest.fixture(scope="module")
def cells_volume() -> np.ndarray:
    """Return single-channel cells3d volume."""
    volume = data.cells3d()
    return volume[:, 1]


CUDA_MANAGER = CUDAManager()


def _gpu_available() -> bool:
    return CUDA_MANAGER.get_num_gpus() > 0


def _middle_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def test_calculate_frc_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    slice_image = _middle_slice(cells_volume)
    cpu_res = calculate_frc(slice_image)

    if not _gpu_available():
        pytest.skip("GPU not available")

    gpu_res = calculate_frc(ascupy(slice_image))
    assert np.isclose(cpu_res, gpu_res, atol=1e-5)


def test_calculate_fsc_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    cpu_res = calculate_fsc(cells_volume)

    if not _gpu_available():
        pytest.skip("GPU not available")

    gpu_res = calculate_fsc(ascupy(cells_volume))
    assert np.allclose(
        [cpu_res["xy"], cpu_res["z"]],
        [gpu_res["xy"], gpu_res["z"]],
        atol=1e-5,
    )


def test_grid_crop_resolution_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    cpu_res = grid_crop_resolution(cells_volume, crop_size=64)

    if not _gpu_available():
        pytest.skip("GPU not available")

    gpu_res = grid_crop_resolution(ascupy(cells_volume), crop_size=64)
    for key in cpu_res:
        assert np.allclose(cpu_res[key], gpu_res[key], atol=1e-5)


def test_five_crop_resolution_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    cpu_res = five_crop_resolution(cells_volume, crop_size=64)

    if not _gpu_available():
        pytest.skip("GPU not available")

    gpu_res = five_crop_resolution(ascupy(cells_volume), crop_size=64)
    for key in cpu_res:
        assert np.allclose(cpu_res[key], gpu_res[key], atol=1e-5)


def test_frc_resolution_difference_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    pad_y = max(0, 512 - cells_volume.shape[1])
    pad_x = max(0, 512 - cells_volume.shape[2])
    pad_width = ((0, 0), (0, pad_y), (0, pad_x))
    padded = np.pad(cells_volume, pad_width, mode="constant")

    cpu_res = frc_resolution_difference(padded, padded)

    if not _gpu_available():
        pytest.skip("GPU not available")

    gpu_res = frc_resolution_difference(ascupy(padded), ascupy(padded))
    assert np.isclose(cpu_res, gpu_res, atol=1e-5)
