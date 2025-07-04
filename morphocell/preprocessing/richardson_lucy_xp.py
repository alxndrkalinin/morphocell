"""Implement the Richardson-Lucy deconvolution algorithm using either NumPy or CuPy.

Modified from https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/deconvolution/richardson_lucy.py

Original license:
--------------------
BSD 3-Clause License

Copyright (c) 2021, True-North-Intelligent-Algorithms
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections.abc import Callable

import numpy as np

from morphocell.cuda import (
    get_array_module,
    check_same_device,
)
from morphocell.skimage import util
from morphocell.image_utils import crop_center, pad_image_to_shape


def richardson_lucy_xp(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 10,
    *,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
    observer_fn: Callable | None = None,
) -> np.ndarray:
    """Lucy-Richardson deconvolution implemented with NumPy or CuPy."""
    xp = get_array_module(image)
    check_same_device(image, psf)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if not noncirc and image.shape != psf.shape:
        psf = pad_image_to_shape(psf, image.shape, mode="constant")

    mask_values = None
    if mask is not None:
        mask = util.img_as_float(mask)
        mask_values = image * (1 - mask)
        image *= mask

    if noncirc:
        orig_size = image.shape
        ext_size = [image.shape[i] + psf.shape[i] - 1 for i in range(image.ndim)]
        psf = pad_image_to_shape(psf, ext_size, mode="constant")

    psf = xp.fft.fftn(xp.fft.ifftshift(psf))
    otf_conj = xp.conjugate(psf)

    if noncirc:
        image = pad_image_to_shape(image, ext_size, mode="constant")
        estimate = xp.full_like(image, xp.mean(image))
    else:
        estimate = image

    if mask is not None:
        htones = xp.ones_like(image) * mask
    else:
        htones = xp.ones_like(image)

    htones = xp.real(xp.fft.ifftn(xp.fft.fftn(htones) * otf_conj))
    htones[htones < 1e-6] = 1

    for i in range(1, n_iter + 1):
        reblurred = xp.real(xp.fft.ifftn(xp.fft.fftn(estimate) * psf))
        ratio = image / (reblurred + 1e-6)
        correction = xp.real(xp.fft.ifftn(xp.fft.fftn(ratio) * otf_conj))

        correction[correction < 0] = 0
        estimate = estimate * correction / htones

        if observer_fn is not None:
            if noncirc:
                unpadded_estimate = crop_center(estimate, orig_size)
                observer_fn(unpadded_estimate, i)
            else:
                observer_fn(estimate, i)

    del psf, otf_conj, htones

    if noncirc:
        estimate = crop_center(estimate, orig_size)

    if mask is not None:
        estimate = estimate * mask + mask_values

    return estimate
