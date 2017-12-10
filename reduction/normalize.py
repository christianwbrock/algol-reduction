import logging
logger = logging.getLogger(__name__)

import numpy as np


def fit_polynomial(xs, ys, deg, ranges):
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
    """

    assert len(xs) == len(ys)
    assert deg > 0
    assert len(ranges) > 0
    assert len(ranges[0]) == 2

    ys = np.asarray(ys)
    xs = np.asarray(xs)

    mask = _ranges_to_mask(xs, ranges)

    ys = ys[mask]
    xs = xs[mask]

    params = np.polyfit(xs, ys, deg)
    logger.debug("polynomial params: %s", params)

    return np.poly1d(params)


def _ranges_to_mask(xs, ranges):
    mask = np.full(len(xs), fill_value=False, dtype=bool)
    for rng in ranges:
        assert len(rng) >= 2

        mask |= [rng[0] <= x <= rng[1] for x in xs]

    return mask


def normalize(xs, ys, deg, ranges):
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
        
    :return: ys divided by the pest fitting polynomial
    """

    assert len(xs) == len(ys)
    assert deg > 0
    # assert ranges.shape[0] > 0
    # assert ranges.shape[1] == 2

    poly = fit_polynomial(xs, ys, deg, ranges)

    array = np.array([ys[i] / poly(xs[i]) for i in range(len(xs))])

    if logger.getEffectiveLevel() <= logging.DEBUG:
        stddev = np.std(array[_ranges_to_mask(xs, ranges)])
        logger.debug("continuum SNR is %.0f", 1.0 / stddev)

    return array


def plot_normalized(filename, ranges, degrees, plot):

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
        poly = fit_polynomial(xs, y1, deg, ranges)
        plot.plot(xs, [poly(x) for x in xs], label="poly_%d" % deg)

        yn = normalize(xs, ys, deg, ranges)
        plot.plot(xs, yn, label="norm_%d" % deg)

