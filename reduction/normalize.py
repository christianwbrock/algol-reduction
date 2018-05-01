""" Normalize spectra
"""
from _weakref import ref

from reduction.spectrum import Spectrum
from reduction.instrument import convolve_with_box, convolve_with_gauss

import numpy as np
from numpy.polynomial.hermite import hermfit, hermval
from matplotlib import pyplot as plt

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)

arg_parser = ArgumentParser(add_help=False)
"""
Use this parser as parent parser in client command line scripts.
"""
arg_parser.add_argument('-r', '--ref', metavar='reference-spectrum',
                        help='An ease source of reference spectra is "The POLLUX Database of Stellar Spectra" '
                             '<http://pollux.graal.univ-montp2.fr>. Also "iSpec" <http://www.blancocuaresma.com/s/iSpec> '
                             'can be used as a frontend for stellar models as SPECTRUM, Turbospectrum, SME, MOOG, '
                             'Synthe/WIDTH9 and others.')
arg_parser.add_argument('-d', '--degree', metavar='polynomial-degree', type=int, default=3,
                        help='degree of the polynomial to be fitted (default: %(default)s)')
arg_parser.add_argument('-c', '--continuum-range', dest='ranges', nargs=2, type=float, metavar=('xmin', 'xmax'),
                        action='append', required=False,
                        help='one or more continuum ranges used for the polynomial fit')
arg_parser.add_argument('--method', choices=['hermit', 'polynomial'], default='polynomial',
                        help='(default:  %(default)s)')
arg_parser.add_argument('--center-minimum', nargs=3, type=float, metavar=('xmin', 'xmax', 'box-size'),
                        help='calculate redshift from plot minimum between xmin and xmax after applying a box filter')
arg_parser.add_argument('--convolve-reference', type=float, metavar='stddev',
                        help='convolve reference spectrum with a gauss kernel to fit the spectrum resolution.')
arg_parser.add_argument('--convolve-spectrum', type=float, metavar='stddev',
                        help='convolve spectrum with a gauss kernel. <HACK>')


def fit_polynomial_args(xs, ys, args):
    return fit_polynomial(xs, ys, args.degree, args.ranges, args.method)


def fit_polynomial(xs, ys, deg, ranges, method):
    """
    return a polynomial of a given degree that best fits the data points 
    passed by xs and ys in the x ranges.
    
    :param xs: array_like, shape(M,)
        x-coordinates of the M sample points ``(xs[i], ys[i])``.
    :param ys: array_like, shape(M,)
        y-coordinates of the M sample points ``(xs[i], ys[i])``.
    :param deg: int
        Degree of the fitting polynomial 
    :param ranges: array_like, shape(N,2)
        At least one x-min, x-max range of points in xs to be used
        for fitting the polynomial.
    :param method: either 'hermit' or 'polynomial' (default: 'polynomial')
    """

    assert len(xs) == len(ys)
    assert deg > 0
    assert len(ranges) > 0
    assert len(ranges[0]) == 2

    assert method in ['hermit', 'polynomial', None]

    logger.debug("fit_polynomial to %d values using '%s' of order %d", len(xs), method, deg)

    ys = np.asarray(ys)
    xs = np.asarray(xs)

    mask = _ranges_to_mask(xs, ranges)
    mask &= np.logical_not(np.isnan(xs))
    mask &= np.logical_not(np.isnan(ys))

    ys = ys[mask]
    xs = xs[mask]

    if method == 'hermit':
        params = hermfit(xs, ys, deg)
    else:
        params = np.polyfit(xs, ys, deg)

    logger.debug("polynomial params: %s", params)

    if method == 'hermit':
        return lambda x: hermval(x, params)
    else:
        return np.poly1d(params)


def _ranges_to_mask(xs, ranges):
    mask = np.full(len(xs), fill_value=False, dtype=bool)
    for rng in ranges:
        assert len(rng) >= 2

        mask |= [rng[0] <= x <= rng[1] for x in xs]

    return mask


def normalize_args(spectrum, args, requested_plot=None, cut=15):
    return normalize_spectrum(spectrum, args.ref, args.degree, args.ranges, args.method, args.center_minimum,
                              args.convolve_spectrum, args.convolve_reference, requested_plot, cut)


def normalize_spectrum(spectrum, ref_spectrum, degree, ranges=None, method=None, center_minimum=None,
                       convolve_spectrum=None, convolve_reference=None, requested_plot=None, cut=15):
    if isinstance(spectrum, str):
        spectrum = Spectrum.load(spectrum)

    if spectrum and convolve_spectrum:
        spectrum = convolve_with_gauss(spectrum, convolve_spectrum)

    if isinstance(ref_spectrum, str):
        ref_spectrum = Spectrum.load(ref_spectrum)

    if ref_spectrum and convolve_reference:
        ref_spectrum = convolve_with_gauss(ref_spectrum, convolve_reference)

    xs = spectrum.xs[cut:-cut]
    ys = spectrum.ys[cut:-cut]

    ys /= np.nanmax(ys)

    if center_minimum and ref_spectrum:
        min_spectrum = _find_minimum(Spectrum.from_arrays(xs, ys), center_minimum)
        min_ref = _find_minimum(ref_spectrum, center_minimum)

        redshift = min_ref - min_spectrum
    else:
        redshift = 0.0

    xs = xs + redshift

    if not ranges:
        ranges = [[xs[0], xs[-1]]]
        if ref_spectrum:
            ranges[0][0] = np.max((ranges[0][0], ref_spectrum.xmin))
            ranges[0][1] = np.min((ranges[0][1], ref_spectrum.xmax))

    ref_ys = ref_spectrum(xs) if ref_spectrum else None

    return normalize(xs, ys, ref_ys, degree, ranges, method, requested_plot)


def _find_minimum(spectrum, range):
    boxed = convolve_with_box(spectrum, range[2])
    mask = [range[0] <= x <= range[1] for x in boxed.xs]
    index = np.nanargmin(boxed.ys[mask])
    return boxed.xs[mask][index]


def normalize(xs, ys, ref_ys, deg, continuum_ranges, method, requested_plot=None):
    """
    :param xs: array_like, shape(M,)
        x-coordinates of the M sample points.
    :param ys: array_like, shape(M,)
        values of the M sample points.
    :param ref_ys: array_like, shape(M,) or None
        reference values of the M sample points.
    :param deg: int
        Degree of the fitting polynomial 
    :param continuum_ranges: array_like, shape(N,2)
        At least one x-min, xmax range of points in xs to be used
        for fitting the polynomial.
    :param method: either 'hermit' or 'polynomial' (default: 'polynomial')
    :param requested_plot: If present, a plot of the normalization is generated.

    :return: ys divided by the pest fitting polynomial and the std-dev
    """

    assert len(xs) == len(ys)
    assert ref_ys is None or len(xs) == len(ref_ys)

    assert deg > 0
    # assert ranges.shape[0] > 0
    # assert ranges.shape[1] == 2

    if ref_ys is not None:
        poly = fit_polynomial(xs, ys / ref_ys, deg, continuum_ranges, method=method)
    else:
        poly = fit_polynomial(xs, ys, deg, continuum_ranges, method=method)

    array = np.array([ys[i] / poly(xs[i]) for i in range(len(xs))])

    mask = _ranges_to_mask(xs, continuum_ranges)
    if ref_ys is not None:
        stddev = np.nanstd((array / ref_ys)[mask])
    else:
        stddev = np.nanstd(array[mask])

    snr = 1.0 / stddev

    logger.debug("continuum SNR is %.0f", snr)

    plot = requested_plot
    if not plot and logger.getEffectiveLevel() < logging.DEBUG:
        fig = plt.figure()
        plot = fig.add_subplot(111)

    if plot:
        xlim = __get_xlim(xs, continuum_ranges, ys, ref_ys)
        plot.set_xlim(xlim)
        plot.set_ylim(__get_ylim(1.3, xlim, xs, ys, ref_ys))

        for r in continuum_ranges:
            plot.axvspan(r[0], r[1], alpha=0.25)

        plot.plot(xs, ys, label='meas')

        if ref_ys is not None:
            plot.plot(xs, ref_ys, label='ref')
            plot.plot(xs, ys / ref_ys, label='meas / ref')

        plot.plot(xs, poly(xs), label='polynomial')
        plot.plot(xs, array, label='normalized')

        plot.legend()

        if not requested_plot:
            plt.show()

    return array, snr


def __get_xlim(xs, ranges, y1, y2):
    min_x = max(xs[0], ranges[0][0])
    max_x = min(xs[-1], ranges[-1][-1])

    for y in [y1, y2]:

        if y is not None:

            for i in range(len(xs)):
                if not np.isnan(y[i]):
                    min_x = max(xs[i], min_x)
                    break

            for i in range(1, len(xs)):
                if not np.isnan(y[-i]):
                    max_x = min(xs[-i], max_x)
                    break

    return min_x, max_x


def __get_ylim(scale, xlim, xs, y1, y2):
    mask = _ranges_to_mask(xs, [xlim])

    min_y = np.min(y1[mask])
    max_y = np.max(y1[mask])

    if y2 is not None:
        min_y = min(min_y, np.min(y2[mask]))
        max_y = max(max_y, np.max(y2[mask]))

    return min_y / scale, max_y * scale
