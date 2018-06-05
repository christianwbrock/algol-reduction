"""
A data sample can be seen as a convolution of a measured quantity with an instrument function.

In this module we assume a Gaussian instrument function
"""

from math import ceil

from astropy.units import Unit
from astropy.convolution import Gaussian1DKernel, Box1DKernel
from astropy.convolution import convolve


def convolve_with_gauss(spectrum, stddev_AA):
    return convolve_with_something(spectrum, stddev_AA, lambda stddev_px, x_size: Gaussian1DKernel(stddev_px, x_size=x_size))


def convolve_with_box(spectrum, stddev_AA):
    return convolve_with_something(spectrum, stddev_AA, lambda stddev_px, x_size: Box1DKernel(stddev_px))


def convolve_with_something(spectrum, stddev_AA, make_kernel):
    from reduction.spectrum import Spectrum

    assert isinstance(spectrum, Spectrum), 'unexpected type {}'.format(type(spectrum))

    if isinstance(stddev_AA, Unit):
        stddev_AA = stddev_AA.to('AA').value

    stddev_px = stddev_AA / spectrum.dx
    x_size = int(ceil(5 * stddev_px))
    if x_size % 2 == 0:
        x_size += 1

    kernel = make_kernel(stddev_px, x_size)

    xs = spectrum.xs
    ys = convolve(spectrum.ys, kernel=kernel, boundary=None)

    assert len(xs) == len(ys)

    # remove convolution boundaries
    clip = kernel.array.size // 2
    xs = xs[clip:-clip]
    ys = ys[clip:-clip]

    assert len(xs) == len(ys)

    return Spectrum.from_arrays(xs, ys)


def deconvolve_with_gauss(xs, ys, sigma):
    raise NotImplemented
