import numpy as np
from numpy.polynomial.hermite import hermfit, hermval

from argparse import ArgumentParser

import warnings
warnings.simplefilter('ignore', np.RankWarning)

import logging
logger = logging.getLogger(__name__)

arg_parser = ArgumentParser(add_help=False)
"""
Use this parser as parent parser in client command line scripts.
"""
arg_parser.add_argument('-d', '--degree', type=int, default=3, help='degree of the polynomial to be fitted')
arg_parser.add_argument('-c', '--continuum-range', dest='ranges', nargs=2, type=float, metavar=('xmin', 'xmax'),
                    action='append', required=True, help='one or more continuum ranges used for the polynomial fit')
arg_parser.add_argument('--method', choices=['hermit', 'polynomial'], default='polynomial')


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


def normalize_args(xs, ys, args):
    return normalize(xs, ys, None, args.degree, args.ranges, args.method)


def normalize(xs, ys, ref_ys, deg, continuum_ranges, method):
    """
    :param ref_ys:
    :param xs: array_like, shape(M,)
        x-coordinates of the M sample points ``(xs[i], ys[i])``.
    :param ys: array_like, shape(M,)
        y-coordinates of the M sample points ``(xs[i], ys[i])``.
    :param deg: int
        Degree of the fitting polynomial 
    :param continuum_ranges: array_like, shape(N,2)
        At least one x-min, xmax range of points in xs to be used
        for fitting the polynomial.
    :param method: either 'hermit' or 'polynomial' (default: 'polynomial')

    :return: ys divided by the pest fitting polynomial
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

    if logger.getEffectiveLevel() <= logging.DEBUG:
        mask = _ranges_to_mask(xs, continuum_ranges)
        if ref_ys is not None:
            stddev = np.nanstd((array / ref_ys)[mask])
        else:
            stddev = np.nanstd(array[mask])

        logger.debug("continuum SNR is %.0f", 1.0 / stddev)

        if logger.getEffectiveLevel() < logging.DEBUG:

            from matplotlib import pyplot as plt

            xlim = __get_xlim(xs, continuum_ranges, ys, ref_ys)
            plt.xlim(xlim)
            plt.ylim(__get_ylim(1.3, xlim, xs, ys, ref_ys))

            for r in continuum_ranges:
                plt.axvspan(r[0], r[1], alpha=0.25)

            plt.plot(xs, ys, label='meas')
            if ref_ys is not None:
                plt.plot(xs, ref_ys, label='ref')
                plt.plot(xs, ys / ref_ys, label='meas / ref')

            plt.plot(xs, poly(xs), label='polynomial')
            plt.plot(xs, array, label='normalized')
            plt.legend()
            plt.show()

    return array


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


def plot_normalized_args(plot, filename, args):
    plot_normalized(plot, filename, args.ranges, [args.degree], args.method)


def plot_normalized(plot, filename, ranges, degrees, method):

    from reduction.spectrum import load

    xs, ys, unit = load(filename)
    xs = xs[15:-15]
    ys = ys[15:-15]

    y1 = 0.6 * ys / ys.max()
    plot.plot(xs, y1, label="orig")
    plot.set_xlabel(unit)

    for r in ranges:
        plot.axvspan(r[0], r[1], alpha=0.25)

    for deg in degrees:
        poly = fit_polynomial(xs, y1, deg, ranges, method=method)
        plot.plot(xs, [poly(x) for x in xs], label="poly_%d" % deg)

        yn = normalize(xs, ys, None, deg, ranges, method=method)
        plot.plot(xs, yn, label="norm_%d" % deg)

