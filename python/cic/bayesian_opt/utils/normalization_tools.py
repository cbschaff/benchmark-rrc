# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch as to
from scipy.special import logsumexp
from typing import Union, Tuple

import pyrado


class UnitCubeProjector:
    """ Project to a unit qube $[0, 1]^d$ and back using explicit bounds """

    def __init__(self, bound_lo: Union[np.ndarray, to.Tensor], bound_up: Union[np.ndarray, to.Tensor]):
        """
        Constructor
        :param bound_lo: lower bound
        :param bound_up: upper bound
        """
        if not type(bound_lo) == type(bound_up):
            raise pyrado.TypeErr(msg="Passed two different types for bounds!")
        if any(bound_lo == pyrado.inf):
            raise pyrado.ValueErr(given=bound_lo, eq_constraint="not +/- inf")
        if any(bound_up == pyrado.inf):
            raise pyrado.ValueErr(given=bound_up, eq_constraint="not +/- inf")

        self.bound_lo = bound_lo
        self.bound_up = bound_up

    def _convert_bounds(
        self, data: Union[to.Tensor, np.ndarray]
    ) -> Union[Tuple[to.Tensor, to.Tensor], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert the bounds into the right type
        :param data: data that is later used for projecting
        :return: bounds casted to the type of data
        """
        if isinstance(data, to.Tensor) and isinstance(self.bound_lo, np.ndarray):
            bound_up, bound_lo = to.from_numpy(self.bound_up), to.from_numpy(self.bound_lo)
        elif isinstance(data, np.ndarray) and isinstance(self.bound_lo, to.Tensor):
            bound_up, bound_lo = self.bound_up.numpy(), self.bound_lo.numpy()
        else:
            bound_up, bound_lo = self.bound_up, self.bound_lo
        return bound_up, bound_lo

    def project_to(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        """
        Normalize every dimension individually using the stored explicit bounds and the L_1 norm.
        :param data: input to project to the unit space
        :return: element of the unit cube
        """
        if not isinstance(data, (to.Tensor, np.ndarray)):
            raise pyrado.TypeErr(given=data, expected_type=[to.Tensor, np.ndarray])

        # Convert if necessary
        bound_up, bound_lo = self._convert_bounds(data)

        span = bound_up - bound_lo
        span[span == 0] = 1.0  # avoid division by 0
        return (data - self.bound_lo) / span

    def project_back(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        """
        Revert the previous normalization using the stored explicit bounds
        :param data: input from the uni space
        :return: element of the original space
        """
        if not isinstance(data, (to.Tensor, np.ndarray)):
            raise pyrado.TypeErr(given=data, expected_type=[to.Tensor, np.ndarray])

        # Convert if necessary
        bound_up, bound_lo = self._convert_bounds(data)

        span = bound_up - bound_lo
        return span * data + bound_lo


def cov(x: to.Tensor, data_along_rows: bool = False):
    """
    Compute the covariance matrix given data.
    .. note::
        Only real valued matrices are supported
    :param x: matrix containing multiple observations of multiple variables
    :param data_along_rows: if `True` the variables are stacked along the columns, else they are along the rows
    :return: covariance matrix given the data
    """
    if x.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if x.dim() < 2:
        x = x.view(1, -1)
    if data_along_rows and x.size(0) != 1:
        # Transpose if necessary
        x = x.t()

    num_samples = x.size(1)
    if num_samples < 2:
        raise pyrado.ShapeErr(msg="Need at least 2 samples to compute the covariance!")

    x -= to.mean(x, dim=1, keepdim=True)
    return x.matmul(x.t()).squeeze() / (num_samples - 1)


def scale_min_max(
    data: Union[np.ndarray, to.Tensor],
    bound_lo: Union[int, float, np.ndarray, to.Tensor],
    bound_up: Union[int, float, np.ndarray, to.Tensor],
) -> Union[np.ndarray, to.Tensor]:
    r"""
    Transform the input data to to be in $[a, b]$.
    :param data: unscaled input ndarray or Tensor
    :param bound_lo: lower bound for the transformed data
    :param bound_up: upper bound for the transformed data
    :return: ndarray or Tensor scaled to be in $[a, b]$
    """
    # Lower bound
    if isinstance(bound_lo, (float, int)) and isinstance(data, np.ndarray):
        bound_lo = bound_lo * np.ones_like(data, dtype=np.float64)
    elif isinstance(bound_lo, (float, int)) and isinstance(data, to.Tensor):
        bound_lo = bound_lo * to.ones_like(data, dtype=to.get_default_dtype())
    elif isinstance(bound_lo, np.ndarray) and isinstance(data, to.Tensor):
        bound_lo = to.from_numpy(bound_lo).to(to.get_default_dtype())
    elif isinstance(bound_lo, to.Tensor) and isinstance(data, np.ndarray):
        bound_lo = bound_lo.numpy()

    # Upper bound
    if isinstance(bound_up, (float, int)) and isinstance(data, np.ndarray):
        bound_up = bound_up * np.ones_like(data, dtype=np.float64)
    elif isinstance(bound_up, (float, int)) and isinstance(data, to.Tensor):
        bound_up = bound_up * to.ones_like(data, dtype=to.get_default_dtype())
    elif isinstance(bound_up, np.ndarray) and isinstance(data, to.Tensor):
        bound_up = to.from_numpy(bound_up).to(to.get_default_dtype())
    elif isinstance(bound_up, to.Tensor) and isinstance(data, np.ndarray):
        bound_up = bound_up.numpy()

    if not (bound_lo < bound_up).all():
        raise pyrado.ValueErr(given_name="lower bound", l_constraint="upper bound")

    if isinstance(data, np.ndarray):
        data_ = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif isinstance(data, to.Tensor):
        data_ = (data - to.min(data)) / (to.max(data) - to.min(data))
    else:
        raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

    return data_ * (bound_up - bound_lo) + bound_lo


def standardize(data: Union[np.ndarray, to.Tensor], eps: float = 1e-8) -> Union[np.ndarray, to.Tensor]:
    r"""
    Standardize the input data to make it $~ N(0, 1)$.
    :param data: input ndarray or Tensor
    :param eps: factor for numerical stability
    :return: standardized ndarray or Tensor
    """
    if isinstance(data, np.ndarray):
        return (data - np.mean(data)) / (np.std(data) + float(eps))
    elif isinstance(data, to.Tensor):
        return (data - to.mean(data)) / (to.std(data) + float(eps))
    else:
        raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])


def normalize(
    x: Union[np.ndarray, to.Tensor], axis: int = -1, order: int = 1, eps: float = 1e-8
) -> Union[np.ndarray, to.Tensor]:
    """
    Normalize a numpy `ndarray` or a PyTroch `Tensor` without changing the input.
    Choosing `axis=1` and `norm_order=1` makes all columns of sum to 1.
    :param x: input to normalize
    :param axis: axis of the array to normalize along
    :param order: order of the norm (e.g., L1 norm: absolute values, L2 norm: quadratic values)
    :param eps: lower bound on the norm, to avoid division by zero
    :return: normalized array
    """
    if isinstance(x, np.ndarray):
        norm_x = np.atleast_1d(np.linalg.norm(x, ord=order, axis=axis))  # calculate norm over axis
        norm_x = np.where(norm_x > eps, norm_x, np.ones_like(norm_x))  # avoid division by 0
        return x / np.expand_dims(norm_x, axis)  # element wise division
    elif isinstance(x, to.Tensor):
        norm_x = to.norm(x, p=order, dim=axis)  # calculate norm over axis
        norm_x = to.where(norm_x > eps, norm_x, to.ones_like(norm_x))  # avoid division by 0
        return x / norm_x.unsqueeze(axis)  # element wise division
    else:
        raise pyrado.TypeErr(given=x, expected_type=[np.array, to.Tensor])


class Standardizer:
    """ A stateful standardizer that remembers the mean and standard deviation for later un-standardization """

    def __init__(self):
        self.mean = None
        self.std = None

    def standardize(self, data: Union[np.ndarray, to.Tensor], eps: float = 1e-8) -> Union[np.ndarray, to.Tensor]:
        r"""
        Standardize the input data to make it $~ N(0, 1)$ and store the input's mean and standard deviation.
        :param data: input ndarray or Tensor
        :param eps: factor for numerical stability
        :return: standardized ndarray or Tensor
        """
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data)
            self.std = np.std(data)
            return (data - self.mean) / (self.std + float(eps))
        elif isinstance(data, to.Tensor):
            self.mean = to.mean(data)
            self.std = to.std(data)
            return (data - self.mean) / (self.std + float(eps))
        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

    def standardize_wo_calculation(self, data: Union[np.ndarray, to.Tensor], eps: float = 1e-8) -> Union[np.ndarray, to.Tensor]:
        r"""
        Standardize the input data to make it $~ N(0, 1)$ and store the input's mean and standard deviation.
        :param data: input ndarray or Tensor
        :param eps: factor for numerical stability
        :return: standardized ndarray or Tensor
        """
        if isinstance(data, np.ndarray):
            return (data - self.mean) / (self.std + float(eps))
        elif isinstance(data, to.Tensor):
            return (data - self.mean) / (self.std + float(eps))
        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

    def unstandardize(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        r"""
        Revert the previous standardization of the input data to make it $~ N(\mu, \sigma)$.
        :param data: input ndarray or Tensor
        :return: un-standardized ndarray or Tensor
        """
        if self.mean is None or self.std is None:
            raise pyrado.ValueErr(msg="Use standardize before unstandardize!")

        # Input type must match stored type
        if isinstance(data, np.ndarray) and isinstance(self.mean, np.ndarray):
            pass
        elif isinstance(data, to.Tensor) and isinstance(self.mean, to.Tensor):
            pass
        elif isinstance(data, np.ndarray) and isinstance(self.mean, to.Tensor):
            self.mean = self.mean.numpy()
            self.std = self.std.numpy()
        elif isinstance(data, to.Tensor) and isinstance(self.mean, np.ndarray):
            self.mean = to.from_numpy(self.mean).to(to.get_default_dtype())
            self.std = to.from_numpy(self.std).to(to.get_default_dtype())

        x_unstd = data * self.std + self.mean
        return x_unstd

    def unstandardize_wo_mean(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        r"""
        Revert the previous standardization of the input data to make it $~ N(\mu, \sigma)$.
        :param data: input ndarray or Tensor
        :return: un-standardized ndarray or Tensor
        """
        if self.mean is None or self.std is None:
            raise pyrado.ValueErr(msg="Use standardize before unstandardize!")

        # Input type must match stored type
        if isinstance(data, np.ndarray) and isinstance(self.mean, np.ndarray):
            pass
        elif isinstance(data, to.Tensor) and isinstance(self.mean, to.Tensor):
            pass
        elif isinstance(data, np.ndarray) and isinstance(self.mean, to.Tensor):
            self.mean = self.mean.numpy()
            self.std = self.std.numpy()
        elif isinstance(data, to.Tensor) and isinstance(self.mean, np.ndarray):
            self.mean = to.from_numpy(self.mean).to(to.get_default_dtype())
            self.std = to.from_numpy(self.std).to(to.get_default_dtype())

        x_unstd = data * self.std
        return x_unstd