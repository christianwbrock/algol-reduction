import numpy as np
import numpy.core.numeric as NX


def fit_polynomial(xs, ys, deg, ranges):
    assert len(xs) == len(ys)
    assert deg > 0
    assert len(ranges) > 0
    assert len(ranges[0]) == 2

    ys = NX.asarray(ys)
    xs = NX.asarray(xs)

    mask = NX.asarray([False for x in xs])

    for range in ranges:
        assert len(range) == 2

        mask |= [range[0] <= x <= range[1] for x in xs]

    ys = ys[mask]
    xs = xs[mask]

    return np.poly1d(np.polyfit(xs, ys, deg))


def normalize(xs, ys, deg, ranges):
    """
    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial

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

    return np.array([ys[i] / poly(xs[i]) for i in range(len(xs))])


def plot_normalized(filename, ranges, degrees, plot):

    from reduction.spectrum import load_from_fit

    xs, ys, unit = load_from_fit(filename)

    y1 = 0.6 * ys / ys.max()
    plot.plot(xs, y1, label="orig")
    plot.set_xlabel(unit)

    face_color = None
    for r in ranges:
        shape = plot.axvspan(r[0], r[1], alpha=0.25)
        face_color = face_color or shape.get_facecolor()
        shape.set_facecolor(face_color)

    for deg in degrees:
        poly = fit_polynomial(xs, y1, deg, ranges)
        plot.plot(xs, [poly(x) for x in xs], label="poly_%d" % (deg))

        yn = normalize(xs, ys, deg, ranges)
        plot.plot(xs, yn, label="norm_%d" % (deg))

