"""
A data sample can be seen as a convolution of a measured quantity with an instrument function.

In this module we assume a Gaussian instrument function
"""

from math import ceil

import numpy as np
from astropy.units import Unit

from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

from reduction.linearinterpolation import LinearInterpolation


def convolve_with_gauss(f, stddev_AA):
    assert isinstance(f, LinearInterpolation), 'unexpected type {}'.format(type(f))

    if isinstance(stddev_AA, Unit):
        stddev_AA = stddev_AA.to('AA').value

    stddev_px = stddev_AA / f.dx
    x_size = int(ceil(5 * stddev_px))
    if x_size % 2 == 0:
        x_size += 1

    kernel = Gaussian1DKernel(stddev_px, x_size=x_size)

    xs = f.xs
    ys = convolve(f.ys, kernel=kernel, boundary='none')

    assert len(xs) == len(ys)

    # remove convolution boundaries
    clip = kernel.array.size // 2
    xs = xs[clip:-clip]
    ys = ys[clip:-clip]

    assert len(xs) == len(ys)

    return LinearInterpolation(xs, ys)

def deconvolute_with_gauss(xs, ys, sigma):
    raise NotImplemented
