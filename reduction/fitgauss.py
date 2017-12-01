import numpy as np
from astropy.modeling import models, fitting

import logging

logger = logging.getLogger(__name__)


def determine_center(xs, ys):
    if not len(xs) == len(ys):
        raise ValueError("list length mismatch")

    mu = 1.0 * sum(xs * ys) / sum(ys)
    sigma = np.sqrt(abs(sum((xs - mu) ** 2 * ys) / sum(ys)))

    logger.info("center=%s, sigma=%s", mu, sigma)
    return mu, sigma


def fit_gauss(xs, ys):
    m, s = determine_center(xs, ys)

    init = models.Gaussian1D(amplitude=ys.max(), mean=m, stddev=s)

    fit = fitting.LevMarLSQFitter()(init, xs, ys)

    logger.info("fit_gauss={0}".format(fit))

    return fit


def fit_lorentz(xs, ys):
    m, s = determine_center(xs, ys)

    init = models.Lorentz1D(amplitude=ys.max(), x_0=m, fwhm=s)

    fit = fitting.LevMarLSQFitter()(init, xs, ys)

    logger.info("fit_lorentz={0}".format(fit))

    return fit


def fit_voigt(xs, ys):
    gauss1d = fit_gauss(xs, ys)

    init = models.Voigt1D(amplitude_L=gauss1d.amplitude, x_0=gauss1d.mean, fwhm_G=gauss1d.stddev, fwhm_L=gauss1d.stddev)

    fit = fitting.LevMarLSQFitter()(init, xs, ys)

    logger.info("fit_lorentz={0}".format(fit))

    return fit
