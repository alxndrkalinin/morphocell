from typing import Any

import numpy as np
import pytest

from morphocell.metrics.frc import calculate_frc, calculate_fsc
from morphocell.cuda import ascupy, CUDAManager
from morphocell.skimage import data


@pytest.fixture(scope="module")
def cells_volume() -> np.ndarray:
    """Return single-channel cells3d volume or skip if unavailable."""
    try:
        volume = data.cells3d()
    except Exception as exc:  # pragma: no cover - dataset may be missing
        pytest.skip(f"cells3d dataset unavailable: {exc}")
    return volume[:, 1]


def _gpu_available() -> bool:
    if not hasattr(_gpu_available, "_cached"):
        _gpu_available._cached = CUDAManager().get_num_gpus() > 0  # type: ignore[attr-defined]
    return _gpu_available._cached  # type: ignore[attr-defined]


def _middle_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def _assert_positive(result: Any) -> None:
    """Recursively assert that result contains positive values."""
    if isinstance(result, dict):
        for val in result.values():
            _assert_positive(val)
    else:
        assert float(result) > 0


def test_calculate_frc_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    slice_image = _middle_slice(cells_volume)
    cpu_res = calculate_frc(slice_image)
    _assert_positive(cpu_res)

    if _gpu_available():
        gpu_res = calculate_frc(ascupy(slice_image))
        assert np.isclose(cpu_res, gpu_res, atol=1e-5)


def test_calculate_fsc_cpu_vs_gpu(cells_volume: np.ndarray) -> None:
    cpu_res = calculate_fsc(cells_volume)
    _assert_positive(cpu_res["xy"])
    _assert_positive(cpu_res["z"])

    if _gpu_available():
        gpu_res = calculate_fsc(ascupy(cells_volume))
        assert np.allclose(
            [cpu_res["xy"], cpu_res["z"]],
            [gpu_res["xy"], gpu_res["z"]],
            atol=1e-5,
        )
