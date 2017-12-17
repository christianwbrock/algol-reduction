import numpy as np
from numpy.polynomial.hermite import hermfit, hermval

from argparse import ArgumentParser

import logging
logger = logging.getLogger(__name__)

arg_parser = ArgumentParser(add_help=False)
"""
Use this parser as parent parser in client command line scripts.
"""
arg_parser.add_argument('--degree', '-d', type=int, default=3, help='degree of the polynomila to be fitted')
arg_parser.add_argument('--continuum-range', '-c', dest='ranges', nargs=2, type=float, metavar=('xmin, xmax'),
                    action='append', required=True, help='continuum range(es) used for the polynomial fit')
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
    return normalize(xs, ys, args.degree, args.ranges, args.method)


def normalize(xs, ys, deg, ranges, method):
    """
    :param xs: array_like, shape(M,)
        x-coordinates of the M sample points ``(xs[i], ys[i])``.
    :param ys: array_like, shape(M,)
        y-coordinates of the M sample points ``(xs[i], ys[i])``.
    :param deg: int
        Degree of the fitting polynomial 
    :param ranges: array_like, shape(N,2)
        At least one x-min, xmax range of points in xs to be used
        for fitting the polynomial.
    :param method: either 'hermit' or 'polynomial' (default: 'polynomial')

    :return: ys divided by the pest fitting polynomial
    """

    assert len(xs) == len(ys)
    assert deg > 0
    # assert ranges.shape[0] > 0
    # assert ranges.shape[1] == 2

    poly = fit_polynomial(xs, ys, deg, ranges, method=method)

    array = np.array([ys[i] / poly(xs[i]) for i in range(len(xs))])

    if logger.getEffectiveLevel() <= logging.DEBUG:
        stddev = np.std(array[_ranges_to_mask(xs, ranges)])
        logger.debug("continuum SNR is %.0f", 1.0 / stddev)

    return array


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

        yn = normalize(xs, ys, deg, ranges, method=method)
        plot.plot(xs, yn, label="norm_%d" % deg)

