# This code is mostly based on the miplib package (https://github.com/sakoho81/miplib),
# licensed as follows:
#
# Copyright (c) 2018, Sami Koho, Molecular Microscopy & Spectroscopy,
# Italian Institute of Technology. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.

# * Neither the name of the Molecular Microscopy and Spectroscopy
# research line, nor the names of its contributors may be used to
# endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER AND CONTRIBUTORS ''AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# In addition to the terms of the license, we ask to acknowledge the use
# of the software in scientific articles by citing:

# Koho, S. et al. Fourier ring correlation simplifies image restoration in fluorescence
# microscopy. Nat. Commun. 10, 3103 (2019).

# Parts of the MIPLIB source code are based on previous BSD licensed
# open source projects:

# pyimagequalityranking:
# Copyright (c) 2015, Sami Koho, Laboratory of Biophysics, University of Turku.
# All rights reserved.

# supertomo:
# Copyright (c) 2014, Sami Koho, Laboratory of Biophysics, University of Turku.
# All rights reserved.

# iocbio-microscope:
# Copyright (c) 2009-2010, Laboratory of Systems Biology, Institute of
# Cybernetics at Tallinn University of Technology. All rights reserved

from math import floor

import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy.interpolate import interp1d, UnivariateSpline

from morphocell.skimage import exposure
from morphocell.image_utils import rotate_image
from argparse import Namespace


def get_frc_options(
    bin_delta: int = 1,
    angle_delta: int = 15,
    extract_angle_delta: float = 0.1,
    resolution_threshold: str = "fixed",
    curve_fit_type: str = "spline",
    disable_hamming: bool = False,
    verbose: bool = False,
) -> Namespace:
    """Return a :class:`argparse.Namespace` with common FRC/FSC parameters."""

    return Namespace(
        d_bin=bin_delta,
        d_angle=angle_delta,
        d_extract_angle=extract_angle_delta,
        disable_hamming=disable_hamming,
        resolution_threshold_criterion=resolution_threshold,
        resolution_threshold_value=0.143,
        resolution_snr_value=7.0,
        frc_curve_fit_degree=3,
        frc_curve_fit_type=curve_fit_type,
        verbose=verbose,
    )


class FixedDictionary(object):
    """
    A dictionary with immutable keys. Is initialized at construction
    with a list of key values.
    """

    def __init__(self, keys):
        assert isinstance(keys, list) or isinstance(keys, tuple)
        self._dictionary = dict.fromkeys(keys)

    def __setitem__(self, key, value):
        if key not in self._dictionary:
            raise KeyError("The key {} is not defined".format(key))
        else:
            self._dictionary[key] = value

    def __getitem__(self, key):
        return self._dictionary[key]

    @property
    def keys(self):
        return list(self._dictionary.keys())

    @property
    def contents(self):
        return list(self._dictionary.keys()), list(self._dictionary.values())


def safe_divide(numerator, denominator):
    """
    Division of numpy arrays that can handle division by zero. NaN results are
    coerced to zero. Also suppresses the division by zero warning.
    :param numerator:
    :param denominator:
    :return:
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result[result == np.inf] = 0.0
        return np.nan_to_num(result)


def cast_to_dtype(data, dtype, rescale=True, remove_outliers=False):
    """
     A function for casting a numpy array into a new data type.
    The .astype() property of Numpy sometimes produces satisfactory
    results, but if the data type to cast into has a more limited
    dynamic range than the original data type, problems may occur.

    :param data:            a np.array object
    :param dtype:           data type string, as in Python
    :param rescale:         switch to enable rescaling pixel
                            values to the new dynamic range.
                            This should always be enabled when
                            scaling to a more limited range,
                            e.g. from float to int
    :param remove_outliers: sometimes deconvolution/fusion generates
                            bright artifacts, which interfere with
                            the rescaling calculation. You can remove them
                            with this switch
    :return:                Returns the input data, cast into the new datatype
    """
    if data.dtype == dtype:
        return data

    if "int" in str(dtype):
        data_info = np.iinfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    elif "float" in str(dtype):
        data_info = np.finfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    else:
        data_max = data.max()
        data_min = data.min()
        print("Warning casting into unknown data type. Detail clippingmay occur")

    # In case of unsigned integers, numbers below zero need to be clipped
    if "uint" in str(dtype):
        data_max = 255
        data_min = 0

    if remove_outliers:
        data = data.clip(0, np.percentile(data, 99.99))

    if rescale is True:
        return rescale_to_min_max(data, data_min, data_max).astype(dtype)
    else:
        return data.clip(data_min, data_max).astype(dtype)


def rescale_to_min_max(data, data_min, data_max):
    """
    A function to rescale data intensities to range, define by
    data_min and data_max input parameters.

    This function is similar to normalize_min_max in image_utils.py but
    takes explicit min/max values instead of calculating percentiles.
    """
    # Rescale data to fit between data_min and data_max
    return exposure.rescale_intensity(data, out_range=(data_min, data_max))


def expand_to_shape(data, shape, dtype=None, background=None):
    """
    Expand data to given shape by zero-padding.
    """
    if dtype is None:
        dtype = data.dtype

    start_index = np.array(shape) - data.shape
    data_start = np.negative(start_index.clip(max=0))
    data = cast_to_dtype(data, dtype, rescale=False)
    if data.ndim == 3:
        data = data[data_start[0] :, data_start[1] :, data_start[2] :]
    else:
        data = data[data_start[0] :, data_start[1] :]

    if background is None:
        background = 0

    if tuple(shape) != data.shape:
        expanded_data = np.zeros(shape, dtype=dtype) + background
        slices = []
        rhs_slices = []
        for s1, s2 in zip(shape, data.shape):
            a, b = (s1 - s2 + 1) // 2, (s1 + s2 + 1) // 2
            c, d = 0, s2
            while a < 0:
                a += 1
                b -= 1
                c += 1
                d -= 1
            slices.append(slice(a, b))
            rhs_slices.append(slice(c, d))
        try:
            expanded_data[tuple(slices)] = data[tuple(rhs_slices)]
        except ValueError:
            print(data.shape, shape)
            raise
        return expanded_data
    else:
        return data


class FourierRingIterator(object):
    """
    A Fourier ring iterator class for 2D images. Calculates a 2D polar coordinate
    centered at the geometric center of the data shape.
    """

    def __init__(self, shape, d_bin):
        """
        :param shape: the volume shape
        :param d_bin: thickness of the ring in pixels
        """

        assert len(shape) == 2

        # Get bin size
        self.d_bin = d_bin
        self.ring_start = 0
        self._nbins = int(floor(shape[0] / (2 * self.d_bin)))
        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        y, x = np.meshgrid(*axes)
        self.meshgrid = (y, x)

        # Create OP vector array
        self.r = np.sqrt(x**2 + y**2)
        # Current ring index
        self.current_ring = self.ring_start

        self.freq_nyq = int(np.floor(shape[0] / 2.0))
        self._radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def radii(self):
        return self._radii

    @property
    def nbins(self):
        return self._nbins

    def get_points_on_ring(self, ring_start, ring_stop):
        arr_inf = self.r >= ring_start
        arr_sup = self.r < ring_stop

        return arr_inf * arr_sup

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(
                self.current_ring * self.d_bin, (self.current_ring + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring - 1


class SectionedFourierRingIterator(FourierRingIterator):
    """
    An iterator for 2D images. Includes the option use only a specific rotated section of
    the fourier ring for FRC calculation.
    """

    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the data
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        FourierRingIterator.__init__(self, shape, d_bin)

        self.d_angle = np.deg2rad(d_angle)

        y, x = self.meshgrid

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, x) + np.pi

        self.phi += self.d_angle / 2
        self.phi[self.phi >= 2 * np.pi] -= 2 * np.pi

        self._angle = 0

        self.angle_sector = self.get_angle_sector(0, d_bin)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        angle = np.deg2rad(value)
        self._angle = angle
        self.angle_sector = self.get_angle_sector(angle, angle + self.d_angle)

    def get_angle_sector(self, phi_min, phi_max):
        """
        Use this to extract
        a section from a sphere that is defined by start and stop angles.

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

    def __getitem__(self, limits):
        """
        Get a single conical section of a 2D ring.

        :param limits:  a list of parameters (ring_start, ring_stop, angle_min, angle_ma)
        that are required to define a single section of a fourier ring.
        """
        (ring_start, ring_stop, angle_min, angle_max) = limits
        ring = self.get_points_on_ring(ring_start, ring_stop)
        cone = self.get_angle_sector(angle_min, angle_max)

        return np.where(ring * cone)

    def __next__(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(
                self.current_ring * self.d_bin, (self.current_ring + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring * self.angle_sector), self.current_ring - 1


class FourierShellIterator(object):
    """
    A Simple Fourier Shell Iterator. Basically the same as a Fourier Ring Iterator,
    but for 3D.
    """

    def __init__(self, shape, d_bin):
        self.d_bin = d_bin

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        z, y, x = np.meshgrid(*axes)
        self.meshgrid = (z, y, x)

        # Create OP vector array
        self.r = np.sqrt(x**2 + y**2 + z**2)

        self.shell_start = 0
        self.shell_stop = int(floor(shape[0] / (2 * self.d_bin))) - 1

        self.current_shell = self.shell_start

        self.freq_nyq = int(np.floor(shape[0] / 2.0))

        self.radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def steps(self):
        return self.radii

    @property
    def nyquist(self):
        return self.freq_nyq

    def get_points_on_shell(self, shell_start, shell_stop):
        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop

        return arr_inf * arr_sup

    def __getitem__(self, limits):
        """
        Get a points on a Fourier shell specified by the start and stop coordinates

        :param shell_start: The start of the shell (0 ... Nyquist)
        :param shell_stop:  The end of the shell

        :return:            Returns the coordinates of the points that are located on
                            the specified shell
        """
        (shell_start, shell_stop) = limits
        shell = self.get_points_on_shell(shell_start, shell_stop)
        return np.where(shell)

    def __iter__(self):
        return self

    def __next__(self):
        shell_idx = self.current_shell

        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_shell += 1

        return np.where(shell), shell_idx


class SectionedFourierShellIterator(FourierShellIterator):
    """
    A sectioned Fourier Shell iterator. Allows dividing a shell into sections, to access
    anisotropic features in the Fourier transform.
    """

    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the data
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        FourierShellIterator.__init__(self, shape, d_bin)

        self.d_angle = np.deg2rad(d_angle)

        z, y, x = self.meshgrid

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, z) + np.pi

        self.phi += self.d_angle / 2
        self.phi[self.phi >= 2 * np.pi] -= 2 * np.pi

        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1
        self.current_rotation = self.rotation_start

        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self):
        return self.radii, self.angles

    def get_angle_sector(self, phi_min, phi_max):
        """
        Assuming a classical spherical coordinate system the azimutahl
        angle is the angle between the x- and y- axes. Use this to extract
        a section from a sphere that is defined by start and stop azimuth
        angles.

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

    def __getitem__(self, limits):
        """
        Get a single section of a 3D shell.

        :param shell_start: The start of the shell (0 ... Nyquist)
        :param shell_stop:  The end of the shell
        :param angle_min:   The start of the section (degrees 0-360)
        :param angle_max:   The end of the section
        :return:            Returns the coordinates of the points that are located inside
                            the portion of a shell that intersects with the points on the
                            cone.
        """
        (shell_start, shell_stop, angle_min, angle_max) = limits
        angle_min = np.deg2rad(angle_min)
        angle_max = np.deg2rad(angle_max)

        shell = self.get_points_on_shell(shell_start, shell_stop)
        cone = self.get_angle_sector(angle_min, angle_max)

        return np.where(shell * cone)

    def __next__(self):
        rotation_idx = self.current_rotation
        shell_idx = self.current_shell

        if rotation_idx <= self.rotation_stop and shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )

            cone = self.get_angle_sector(
                self.current_rotation * self.d_angle,
                (self.current_rotation + 1) * self.d_angle,
            )
        else:
            raise StopIteration

        if rotation_idx >= self.rotation_stop:
            self.current_rotation = 0
            self.current_shell += 1
        else:
            self.current_rotation += 1

        return np.where(shell * cone), shell_idx, rotation_idx


class HollowSectionedFourierShellIterator(SectionedFourierShellIterator):
    """
    A sectioned Fourier shell iterator with the added possibility to remove
    a central section of the cone, to better deal with interpolation artefacts etc.
    """

    def __init__(self, shape, d_bin, d_angle, d_extract_angle=5):
        SectionedFourierShellIterator.__init__(self, shape, d_bin, d_angle)

        self.d_extract_angle = np.deg2rad(d_extract_angle)

    def get_angle_sector(self, phi_min, phi_max):
        """
        Assuming a classical spherical coordinate system the azimutahl
        angle is the angle between the x- and y- axes. Use this to extract
        a section from a sphere that is defined by start and stop azimuth
        angles.

        In the hollow implementation a small slice in the center of the section is
        removed to avoid the effect of resampling when calculating the resolution
        along the lowest resolution axis (z), on images with very isotropic resolution
        (e.g. STED).

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        # Calculate angular sector
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        full_section = arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

        # Calculate part of the section to exclude
        sector_center = phi_min + (phi_max - phi_min) / 2
        phi_min_ext = sector_center - self.d_extract_angle
        phi_max_ext = sector_center + self.d_extract_angle

        arr_inf_ext = self.phi >= phi_min_ext
        arr_sup_ext = self.phi < phi_max_ext

        arr_inf_neg_ext = self.phi >= phi_min_ext + np.pi
        arr_sup_neg_ext = self.phi < phi_max_ext + np.pi

        extract_section = arr_inf_ext * arr_sup_ext + arr_inf_neg_ext * arr_sup_neg_ext

        return np.logical_xor(full_section, extract_section)


class AxialExcludeSectionedFourierShellIterator(HollowSectionedFourierShellIterator):
    """
    A sectioned Fourier shell iterator with the added possibility to remove
    a central section of the cone, to better deal with interpolation artefacts etc.
    """

    def __init__(self, shape, d_bin, d_angle, d_extract_angle=5):
        HollowSectionedFourierShellIterator.__init__(self, shape, d_bin, d_angle)

    def get_angle_sector(self, phi_min, phi_max):
        """
        Assuming a classical spherical coordinate system the azimutahl
        angle is the angle between the x- and y- axes. Use this to extract
        a section from a sphere that is defined by start and stop azimuth
        angles.

        In the hollow implementation a small slice in the center of the section is
        removed to avoid the effect of resampling when calculating the resolution
        along the lowest resolution axis (z), on images with very isotropic resolution
        (e.g. STED).

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        # Calculate angular sector
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        full_section = arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

        axis_pos = np.deg2rad(90) + self.d_angle / 2
        axis_neg = np.deg2rad(270) + self.d_angle / 2

        if phi_min <= axis_pos <= phi_max:
            phi_min_ext = axis_pos - self.d_extract_angle
            phi_max_ext = axis_pos + self.d_extract_angle

        elif phi_min <= axis_neg <= phi_max:
            # Calculate part of the section to exclude
            phi_min_ext = axis_neg - self.d_extract_angle
            phi_max_ext = axis_neg + self.d_extract_angle

        else:
            return full_section

        arr_inf_ext = self.phi >= phi_min_ext
        arr_sup_ext = self.phi < phi_max_ext

        arr_inf_neg_ext = self.phi >= phi_min_ext + np.pi
        arr_sup_neg_ext = self.phi < phi_max_ext + np.pi

        extract_section = arr_inf_ext * arr_sup_ext + arr_inf_neg_ext * arr_sup_neg_ext

        return np.logical_xor(full_section, extract_section)


class RotatingFourierShellIterator(FourierShellIterator):
    """
    A 3D Fourier Ring Iterator -- not a Fourier Shell Iterator, but rather
    single planes are extracted from a 3D shape by rotating the XY plane,
    as in:

    Nieuwenhuizen, Rpj, K. A. Lidke, and Mark Bates. 2013.
    "Measuring Image Resolution in Optical Nanoscopy." Nature
    advance on (April). https://doi.org/10.1038/nmeth.2448.

    Here the shell iteration is still in 3D (for compatilibility with the others
    which doesn't make much sense in terms of calculation effort, but it should make
    it possible to

    """

    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the data
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        assert len(shape) == 3, "This iterator assumes a 3D shape"

        FourierShellIterator.__init__(self, shape, d_bin)

        plane = expand_to_shape(np.ones((1, shape[1], shape[2])), shape)

        self.plane = plane

        self.rotated_plane = plane > 0

        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1
        self.current_rotation = self.rotation_start

        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self):
        return self.radii, self.angles

    def __getitem__(self, limits):
        """
        Get a single section of a 3D shell.

        :param shell_start: The start of the shell (0 ... Nyquist)
        :param shell_stop:  The end of the shell
        :param angle:
        """
        (shell_start, shell_stop, angle) = limits
        rotated_plane = rotate_image(self.plane, angle)

        points_on_plane = rotated_plane > 0
        points_on_shell = self.get_points_on_shell(shell_start, shell_stop)

        return np.where(points_on_plane * points_on_shell)

    def __next__(self):
        rotation_idx = self.current_rotation + 1
        shell_idx = self.current_shell

        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
            self.current_shell += 1

        elif rotation_idx <= self.rotation_stop:
            rotated_plane = rotate_image(
                self.plane, self.angles[rotation_idx], interpolation="linear"
            )

            self.rotated_plane = rotated_plane > 0
            self.current_shell = 0
            shell_idx = 0
            self.current_rotation += 1

            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )

        else:
            raise StopIteration

        return np.where(shell * self.rotated_plane), shell_idx, self.current_rotation


class FourierCorrelationDataCollection(object):
    """
    A container for the directional Fourier correlation data
    """

    def __init__(self):
        self._data = {}

        self.iter_index = 0

    def __setitem__(self, key, value):
        assert isinstance(key, (int, np.integer))
        assert isinstance(value, FourierCorrelationData)

        self._data[str(key)] = value

    def __getitem__(self, key):
        return self._data[str(key)]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = list(self._data.items())[self.iter_index]
        except IndexError:
            self.iter_index = 0
            raise StopIteration

        self.iter_index += 1
        return item

    def __len__(self):
        return len(self._data)

    def clear(self):
        self._data.clear()

    def items(self):
        return list(self._data.items())

    def nitems(self):
        return len(self._data)

    def as_dataframe(self, include_results=False):
        """
        Convert a FourierCorrelationDataCollection object into a Pandas
        dataframe. Only returns the raw Fourier correlation data,
        not the processed results.

        :return: A dataframe with columns: Angle (categorical), Correlation (Y),
                 Frequency (X) and nPoints (number of points in each bin)
        """
        df = pd.DataFrame(columns=["Correlation", "Frequency", "nPoints", "Angle"])

        for key, dataset in self._data.items():
            df_temp = dataset.as_dataframe(include_results=include_results)

            angle = np.full(len(df_temp), int(key), dtype=np.int64)
            df_temp["Angle"] = angle

            df = pd.concat([df, df_temp], ignore_index=True)

        df["Angle"] = df["Angle"].astype("category")
        return df


class FourierCorrelationData(object):
    """
    A datatype for FRC data

    """

    # todo: the dictionary format here is a bit clumsy. Maybe change to a simpler structure

    def __init__(self, data=None):
        correlation_keys = (
            "correlation frequency points-x-bin curve-fit curve-fit-coefficients"
        )
        resolution_keys = (
            "threshold criterion resolution-point "
            "resolution-threshold-coefficients resolution spacing"
        )

        self.resolution = FixedDictionary(resolution_keys.split())
        self.correlation = FixedDictionary(correlation_keys.split())

        if data is not None:
            assert isinstance(data, dict)

            for key, value in data.items():
                if key in self.resolution.keys:
                    self.resolution[key] = value
                elif key in self.correlation.keys:
                    self.correlation[key] = value
                else:
                    raise ValueError("Unknown key found in the initialization data")

    def as_dataframe(self, include_results=False):
        """
        Convert a FourierCorrelationData object into a Pandas
        dataframe. Only returns the raw Fourier correlation data,
        not the processed results.

        :return: A dataframe with columns: Correlation (Y), Frequency (X) and
                 nPoints (number of points in each bin)
        """
        if include_results is False:
            to_df = {
                "Correlation": self.correlation["correlation"],
                "Frequency": self.correlation["frequency"],
                "nPoints": self.correlation["points-x-bin"],
            }
        else:
            resolution = np.full(
                self.correlation["correlation"].shape,
                self.resolution["resolution"],
                dtype=np.float32,
            )
            resolution_point_x = np.full(
                self.correlation["correlation"].shape,
                self.resolution["resolution-point"][0],
                dtype=np.float32,
            )
            resolution_point_y = np.full(
                self.correlation["correlation"].shape,
                self.resolution["resolution-point"][1],
                dtype=np.float32,
            )
            threshold = (self.resolution["threshold"],)

            to_df = {
                "Correlation": self.correlation["correlation"],
                "Frequency": self.correlation["frequency"],
                "nPoints": self.correlation["points-x-bin"],
                "Resolution": resolution,
                "Resolution_X": resolution_point_x,
                "Resolution_Y": resolution_point_y,
                "Threshold": threshold,
            }

        return pd.DataFrame(to_df)


def fit_frc_curve(data_set, degree, fit_type="spline"):
    """
    Calculate a least squares curve fit to the FRC Data
    :return: None. Will modify the frc argument in place
    """
    assert isinstance(data_set, FourierCorrelationData)

    data = data_set.correlation["correlation"]

    if fit_type == "smooth-spline":
        equation = UnivariateSpline(data_set.correlation["frequency"], data)
        equation.set_smoothing_factor(0.25)
        # equation = interp1d(data_set.correlation["frequency"],
        #                     data, kind='slinear')

    elif fit_type == "spline":
        equation = interp1d(data_set.correlation["frequency"], data, kind="slinear")

    elif fit_type == "polynomial":
        coeff = np.polyfit(
            data_set.correlation["frequency"],
            data,
            degree,
            w=1 - data_set.correlation["frequency"] ** 3,
        )
        equation = np.poly1d(coeff)
    else:
        raise AttributeError(fit_type)

    data_set.correlation["curve-fit"] = equation(data_set.correlation["frequency"])

    return equation


def calculate_snr_threshold_value(points_x_bin, snr):
    """
    A function to calculate a SNR based resolution threshold, as described
    in ...

    :param points_x_bin: a 1D Array containing the numbers of points at each
    FRC/FSC ring/shell
    :param snr: the expected SNR value
    :return:
    """
    nominator = snr + safe_divide(2.0 * np.sqrt(snr) + 1, np.sqrt(points_x_bin))
    denominator = snr + 1 + safe_divide(2.0 * np.sqrt(snr), np.sqrt(points_x_bin))
    return safe_divide(nominator, denominator)


def calculate_resolution_threshold_curve(data_set, criterion, threshold, snr):
    """
    Calculate the two sigma curve. The FRC should be run first, as the results of the two sigma
    depend on the number of points on the fourier rings.

    :return:  Adds the
    """
    assert isinstance(data_set, FourierCorrelationData)

    points_x_bin = data_set.correlation["points-x-bin"]

    if points_x_bin[-1] == 0:
        points_x_bin[-1] = points_x_bin[-2]

    if criterion == "one-bit":
        nominator = 0.5 + safe_divide(2.4142, np.sqrt(points_x_bin))
        denominator = 1.5 + safe_divide(1.4142, np.sqrt(points_x_bin))
        points = safe_divide(nominator, denominator)

    elif criterion == "half-bit":
        nominator = 0.2071 + safe_divide(1.9102, np.sqrt(points_x_bin))
        denominator = 1.2071 + safe_divide(0.9102, np.sqrt(points_x_bin))
        points = safe_divide(nominator, denominator)

    elif criterion == "three-sigma":
        points = safe_divide(
            np.full(points_x_bin.shape, 3.0), (np.sqrt(points_x_bin) + 3.0 - 1)
        )

    elif criterion == "fixed":
        points = threshold * np.ones(len(data_set.correlation["points-x-bin"]))
    elif criterion == "snr":
        points = calculate_snr_threshold_value(points_x_bin, snr)

    else:
        raise AttributeError()

    if criterion != "fixed":
        # coeff = np.polyfit(data_set.correlation["frequency"], points, 3)
        # equation = np.poly1d(coeff)
        equation = interp1d(data_set.correlation["frequency"], points, kind="slinear")
        curve = equation(data_set.correlation["frequency"])
    else:
        curve = points
        equation = None

    data_set.resolution["threshold"] = curve
    return equation


class FourierCorrelationAnalysis(object):
    def __init__(self, data, spacing, args):
        assert isinstance(data, FourierCorrelationDataCollection)

        self.data_collection = data
        self.args = args
        self.spacing = spacing

    def execute(self, z_correction=1):
        """
        Calculate the spatial resolution as a cross-section of the FRC and Two-sigma curves.

        :return: Returns the calculation results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        criterion = self.args.resolution_threshold_criterion
        threshold = self.args.resolution_threshold_value
        snr = self.args.resolution_snr_value
        # tolerance = self.args.resolution_point_sigma
        degree = self.args.frc_curve_fit_degree
        fit_type = self.args.frc_curve_fit_type
        verbose = self.args.verbose

        def pdiff1(x):
            return abs(frc_eq(x) - two_sigma_eq(x))

        def pdiff2(x):
            return abs(frc_eq(x) - threshold)

        def first_guess(x, y, threshold):
            # y_smooth = savgol_filter(y, 5, 2)
            # return x[np.argmin(np.abs(y_smooth - threshold))]

            difference = y - threshold

            return x[np.where(difference <= 0)[0][0] - 1]
            # return x[np.argmin(np.abs(y - threshold))]

        for key, data_set in self.data_collection:
            if verbose:
                print("Calculating resolution point for dataset {}".format(key))
            frc_eq = fit_frc_curve(data_set, degree, fit_type)
            two_sigma_eq = calculate_resolution_threshold_curve(
                data_set, criterion, threshold, snr
            )

            """
            Todo: Make the first quess adaptive. For example find the data point at which FRC
            value is closest to the mean of the threshold
            """

            # Find intersection
            fit_start = first_guess(
                data_set.correlation["frequency"],
                data_set.correlation["correlation"],
                np.mean(data_set.resolution["threshold"]),
            )
            if self.args.verbose:
                print("Fit starts at {}".format(fit_start))
                disp = 1
            else:
                disp = 0
            root = optimize.fmin(
                pdiff2 if criterion == "fixed" else pdiff1, fit_start, disp=disp
            )[0]
            data_set.resolution["resolution-point"] = (frc_eq(root), root)
            data_set.resolution["criterion"] = criterion

            angle = np.deg2rad(int(key))
            z_correction_multiplier = 1 + (z_correction - 1) * np.abs(np.sin(angle))
            resolution = z_correction_multiplier * (2 * self.spacing / root)

            data_set.resolution["resolution"] = resolution
            data_set.resolution["spacing"] = self.spacing * z_correction_multiplier

            self.data_collection[int(key)] = data_set

            # # # Find intersection
            # root, result = optimize.brentq(
            #     pdiff2 if criterion == 'fixed' else pdiff1,
            #     0.0, 1.0, xtol=tolerance, full_output=True)
            #
            # # Save result, if intersection was found
            # if result.converged is True:
            #     data_set.resolution["resolution-point"] = (frc_eq(root), root)
            #     data_set.resolution["criterion"] = criterion
            #     resolution = 2 * self.spacing / root
            #     data_set.resolution["resolution"] = resolution
            #     self.data_collection[int(key)] = data_set
            # else:
            #     print "Could not find an intersection for the curves for the dataset %s." % key

        return self.data_collection
