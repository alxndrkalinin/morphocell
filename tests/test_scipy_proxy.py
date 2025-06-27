import numpy as np

from morphocell.cuda import CUDAManager, ascupy, asnumpy
import morphocell.scipy as mc_scipy


def _gpu_available() -> bool:
    return CUDAManager().get_num_gpus() > 0


def test_ndimage_laplace_cpu_vs_gpu() -> None:
    a = np.asarray([1, 2, 3], dtype=float)
    cpu_res = mc_scipy.ndimage.laplace(a)

    if _gpu_available():
        gpu_res = mc_scipy.ndimage.laplace(ascupy(a))
        assert np.allclose(cpu_res, asnumpy(gpu_res))
    else:
        gpu_res = mc_scipy.ndimage.laplace(a)
        assert np.allclose(cpu_res, gpu_res)
