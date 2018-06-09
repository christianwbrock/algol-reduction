""" Normalize spectra
"""
import logging
from argparse import ArgumentParser

import numpy as np
from numpy.polynomial.hermite import hermfit, hermval
from matplotlib import pyplot as plt

from reduction.spectrum import Spectrum, find_minimum
from reduction.instrument import convolve_with_gauss
from reduction.utils.ranges import closed_range, LebesgueSet


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
arg_parser.add_argument('-C', '--non-continuum-range', dest='non_ranges', nargs=2, type=float, metavar=('xmin', 'xmax'),
                        action='append', required=False,
                        help='one or more ranges not in the continuum used for the polynomial fit')
arg_parser.add_argument('--method', choices=['hermit', 'polynomial'], default='polynomial',
                        help='(default:  %(default)s)')
arg_parser.add_argument('--center-minimum', nargs=3, type=float, metavar=('xmin', 'xmax', 'box-size'),
                        help='calculate redshift from plot minimum between xmin and xmax after applying a box filter')
arg_parser.add_argument('--convolve-reference', type=float, metavar='stddev',
                        help='convolve reference spectrum with a gauss kernel to fit the spectrum resolution.')
arg_parser.add_argument('--convolve-spectrum', type=float, metavar='stddev',
                        help='convolve spectrum with a gauss kernel. <HACK>')


def normalize_args(spectrum, args, requested_plot=None, requested_spectra=None, cut=15):
    """
    Normalize spectrum using commandline args from `arg_arg_parser`.

    Parameters
    ----------
        spectrum: Spectrum
            the spectrum to normalize

        args : dict
            commandline args from `arg_arg_parser`

        requested_plot : pyplot.axes or None
            If not None, used to plot plot normalization

        requested_spectra : dict or None
            If not None, several intermediate results are stored here.

        cut : int
            the first and last `cut` values of the spectrum are discarded

    Returns
    -------
        norm, snr: tupel
            normalization result and the calculated SNR
    """
    return _normalize_spectrum(spectrum, args.ref, args.degree, args.ranges, args.non_ranges, args.method,
                               args.center_minimum, args.convolve_spectrum, args.convolve_reference, requested_plot,
                               requested_spectra, cut)


def _normalize_spectrum(spectrum, ref_spectrum, degree, ranges=None, non_ranges=None, method=None, center_minimum=None,
                        convolve_spectrum=None, convolve_reference=None, requested_plot=None, requested_spectra=None,
                        cut=15):
    if isinstance(spectrum, str):
        spectrum = Spectrum.load(spectrum)

    if spectrum and convolve_spectrum and convolve_spectrum > 0.0:
        spectrum = convolve_with_gauss(spectrum, convolve_spectrum)

    if isinstance(ref_spectrum, str):
        ref_spectrum = Spectrum.load(ref_spectrum)

    spectrum_resolution = spectrum.resolution

    if cut and cut > 0:
        spectrum = Spectrum.from_arrays(spectrum.xs[cut:-cut], spectrum.ys[cut:-cut], spectrum.filename)

    if ref_spectrum:
        if convolve_reference:
            if convolve_reference > 0.0:
                ref_spectrum = convolve_with_gauss(ref_spectrum, convolve_reference)
        elif spectrum_resolution:
            convolve_reference = 0.5 * (ref_spectrum.xmax + ref_spectrum.xmin) / spectrum_resolution / 2.354
            ref_spectrum = convolve_with_gauss(ref_spectrum, convolve_reference)

    xs = spectrum.xs
    ys = spectrum.ys

    ys /= np.nanmax(ys)

    if center_minimum and ref_spectrum:
        min_spectrum = find_minimum(Spectrum.from_arrays(xs, ys), *center_minimum)
        min_ref = find_minimum(ref_spectrum, *center_minimum)

        redshift = min_ref - min_spectrum
    else:
        redshift = 0.0

    xs = xs + redshift

    continuum_ranges = closed_range(np.nanmin(xs), np.nanmax(xs))
    if ref_spectrum:
        continuum_ranges &= closed_range(ref_spectrum.xmin, ref_spectrum.xmax)

    if ranges:
        continuum_ranges &= _list_to_set(ranges)

    if non_ranges:
        continuum_ranges &= ~ _list_to_set(non_ranges)

    ref_ys = ref_spectrum(xs) if ref_spectrum else None

    return normalize(xs, ys, ref_ys, degree, continuum_ranges, method, requested_plot, requested_spectra)


def fit_polynomial(xs, ys, deg, continuum_ranges, method):
    """
    Return a polynomial of a given degree that best fits the data points
    passed by xs and ys in the x ranges.

    Parameters
    ----------
        xs: array_like, shape(M,)
            x-coordinates of the M sample points ``(xs[i], ys[i])``.

        ys: array_like, shape(M,)
            y-coordinates of the M sample points ``(xs[i], ys[i])``.

        deg: int
            Degree of the fitting polynomial

        continuum_ranges: LebesgueSet
            Defines ranges belonging top the continuum

        method: either 'hermit' or 'polynomial' (default: 'polynomial')

    Returns
    -------
        polynomial : callable
    """

    assert len(xs) == len(ys)
    assert deg > 0
    assert not continuum_ranges or isinstance(continuum_ranges, LebesgueSet)

    assert method in ['hermit', 'polynomial', None]

    logger.debug("fit_polynomial to %d values using '%s' of order %d", len(xs), method, deg)

    ys = np.asarray(ys)
    xs = np.asarray(xs)

    mask = _ranges_to_mask(xs, continuum_ranges)
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
    if ranges:
        assert isinstance(ranges, LebesgueSet)
        return np.array([x in ranges for x in xs])

    else:
        return np.full(len(xs), fill_value=True, dtype=bool)


def _list_to_set(lst):
    """
    convert a list of range boundaries to a LebesgueSet
    """

    if isinstance(lst, LebesgueSet):
        return lst

    assert not lst or np.ndim(lst) == 2
    assert not lst or np.shape(lst)[1] >= 2

    result = None

    for item in lst:
        item = closed_range(item[0], item[1])
        result = result.union(item) if result else item

    return result


def normalize(xs, ys, ref_ys, deg, continuum_ranges, method=None, requested_plot=None, requested_spectra=None):
    """
    Return a polynomial of a given degree that best fits the data points
    passed by xs and ys in the x ranges.

    Parameters
    ----------
        xs: array_like, shape(M,)
            x-coordinates of the M sample points ``(xs[i], ys[i])``.

        ys: array_like, shape(M,)
            y-coordinates of the M sample points ``(xs[i], ys[i])``.

        ref_ys: array_like, shape(M,) or None
            reference values of the M sample points.

        deg: int
            Degree of the fitting polynomial

        continuum_ranges: LebesgueSet
            Defines ranges belonging top the continuum

        method: either 'hermit' or 'polynomial' (default: 'polynomial')

        requested_plot : pyplot.axes or None
            If not None, used to plot plot normalization

        requested_spectra : dict or None
            If not None, several intermediate results are stored here.

    Returns
    -------
        norm, snr: array_like, shape(M,), float
            normalization result and the calculated SNR
    """

    assert len(xs) == len(ys)
    assert ref_ys is None or len(xs) == len(ref_ys)

    assert continuum_ranges is None or isinstance(continuum_ranges, LebesgueSet)

    assert deg > 0
    # assert ranges.shape[0] > 0
    # assert ranges.shape[1] == 2

    if ref_ys is not None:
        poly = fit_polynomial(xs, ys / ref_ys, deg, continuum_ranges, method=method)
    else:
        poly = fit_polynomial(xs, ys, deg, continuum_ranges, method=method)

    norm = np.array([ys[i] / poly(xs[i]) for i in range(len(xs))])

    mask = _ranges_to_mask(xs, continuum_ranges)
    if ref_ys is not None:
        stddev = np.nanstd((norm / ref_ys)[mask])
    else:
        stddev = np.nanstd(norm[mask])

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

        if continuum_ranges and continuum_ranges.is_bounded():
            for r in continuum_ranges.intervals():
                plot.axvspan(r[0], r[1], alpha=0.25)

        plot.plot(xs, ys, label='meas')

        if ref_ys is not None:
            plot.plot(xs, ref_ys, label='ref')
            plot.plot(xs, ys / ref_ys, label='meas / ref')

        plot.plot(xs, poly(xs), label='polynomial')
        plot.plot(xs, norm, label='normalized')

        plot.legend()

        if not requested_plot:
            plt.show()
    
    if requested_spectra is not None:
        requested_spectra['xs'] = xs
        requested_spectra['ys'] = ys
        requested_spectra['ref_ys'] = ref_ys
        requested_spectra['norm'] = norm
        requested_spectra['fit'] = poly(xs)

    return norm, snr


def __get_xlim(xs, ranges, y1, y2):
    min_x = max(xs[0], ranges.lower_bound())
    max_x = min(xs[-1], ranges.upper_bound())

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

    mask = _ranges_to_mask(xs, closed_range(*xlim))

    min_y = np.min(y1[mask])
    max_y = np.max(y1[mask])

    if y2 is not None:
        min_y = min(min_y, np.min(y2[mask]))
        max_y = max(max_y, np.max(y2[mask]))

    return min_y / scale, max_y * scale
