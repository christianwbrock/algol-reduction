#!python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.commandline import poly_glob, filename_parser, verbose_parser, get_loglevel
from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.spectrum import Spectrum
from reduction.stars.algol import Algol, algol_coordinate
from reduction.constants import H_ALPHA
from reduction.normalize import normalize

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation

from astropy.convolution import Box1DKernel
from astropy.convolution import convolve

from astropy.modeling.fitting import SimplexLSQFitter
from reduction.nan_statistics import nan_leastsquare

from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np

import os
import os.path

from collections import namedtuple

import logging

logger = logging.getLogger(__name__)

Diff = namedtuple('Diff', 'wavelength diff phase')


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            description='Generate LaTeX report displaying spectrums normalized around the Halpha line.')
    parser.add_argument('--fit-sigma', action='store_true',
                        help='Modify model stddev to fit best data')
    parser.add_argument('--fit-redshift', action='store_true',
                        help='Modify model redshift to best fit data')

    parser.add_argument('--deg', type=int, default=5,
                        help='Degree of the normalization polynomial (default: %(default)s)')

    parser.add_argument('--cmap', default='bwr',
                        help='A valid matplotlib colormap name (default: %(default)s)')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    if args.cmap not in cm.datad.keys():
        logger.warning('Invalid colormap not in %s', cm.datad.keys())
        args.cmap = parser.get_default('cmap')

    observer_location = EarthLocation.from_geodetic(lon=15.0, lat=50.0)
    algol = Algol()

    padding = 10.0
    continuum_ranges = ((6520, H_ALPHA.value - padding), (H_ALPHA.value + padding, 6610))

    # list of Diffs
    diff_plots = []

    filenames = poly_glob(args.filenames)

    spectra = []
    for n, filename in enumerate(filenames, start=1):

        logger.info("pass1 %d/%d: %s", n, len(filenames), filename)

        for spectrum in Spectrum.load(filename, slice(None)):

            obs_time = spectrum.obs_date
            if not obs_time:
                logger.error("%s has no observation date", spectrum.filename)
                continue

            spectra.append(spectrum)

    for n, spectrum in enumerate(sorted(spectra, key=lambda sp: (sp.observer, sp.obs_date)), start=1):

        logger.info("pass2 %d/%d: %s", n, len(spectra), spectrum.short_name)

        xs = spectrum.xs
        ys = spectrum.ys

        xs = xs[15:-15]
        ys = ys[15:-15]

        ys = ys / ys.max()

        obs_time = spectrum.obs_date
        res = spectrum.resolution

        light_travel_time = obs_time.light_travel_time(algol_coordinate, location=observer_location)
        obs_time -= light_travel_time
        algol_rv_a = algol.rv_A(obs_time)
        radial_velocity_correction = algol_coordinate.radial_velocity_correction(obstime=obs_time,
                                                                                 location=observer_location)
        #  rv_predicted = algol_rv_a - radial_velocity_correction
        phase = algol.AB.phase(obs_time)

        redshift_from_data = u.Quantity(_find_minimum(xs, ys, spectrum.dx, 10, 1.5), u.AA) - H_ALPHA

        sigma = H_ALPHA / (res or 15000)

        initial_model = AlgolHAlphaModel(redshift=redshift_from_data, sigma=sigma)
        initial_model.scale.fixed = True
        initial_model.redshift.fixed = not args.fit_redshift
        initial_model.sigma.fixed = not args.fit_sigma

        normalized, snr = normalize(xs, ys, ref_ys=initial_model(xs), deg=args.deg, continuum_ranges=continuum_ranges,
                               method=None)

        improve_model = args.fit_sigma or args.fit_redshift
        if improve_model:

            fitter = SimplexLSQFitter()
            fitter._stat_method = nan_leastsquare
            fitter._opt_method._maxiter = 2000

            final_model = fitter(initial_model, xs, normalized, weights=np.sqrt(normalized))

            logger.debug("fit info: %s", fitter.fit_info)

            normalized, snr = normalize(xs, ys, ref_ys=final_model(xs), deg=args.deg, continuum_ranges=continuum_ranges,
                                   method=None)
        else:
            final_model = initial_model

        diff_plots.append(Diff(xs - final_model.redshift, normalized - final_model(xs), phase))

        if logger.getEffectiveLevel() <= logging.DEBUG:

            fig = plt.figure()
            plot = fig.add_subplot(111)
            plot.set_ylim(-0.5, 1.5)

            xlim = np.asarray(final_model.get_xlimits())
            xlim[0] = max(xlim[0], min(*continuum_ranges[:][0]))
            xlim[1] = min(xlim[1], max(*continuum_ranges[:][1]))
            plot.set_xlim(xlim)

            plot.plot(xs, 0.6 * ys, label='measured')
            plot.plot(xs, normalized, label='normalized')
            plot.plot(xs, initial_model(xs), label='predicted %s' % initial_model)
            if improve_model:
                plot.plot(xs, final_model(xs), label='fitted %s' % final_model)
            plot.plot(xs, normalized - final_model(xs), label='normalized - fitted')

            plot.set_title('%s -- phase = %.2f' % (spectrum.short_name, phase))

            plot.legend(loc='upper right')
            plt.show()

    # end filenames

    if len(diff_plots):

        fig, plot = plt.subplots()
        plot.set_ylim(-0.6, +1.6)
        plot.set_ylabel('Phase')
        plot.set_xlabel('$Wavelength (\AA)$')
        plot.set_title('measured - predicted')

        vmin = np.min([np.nanmin(diff.diff) for diff in diff_plots])
        vmax = np.max([np.nanmax(diff.diff) for diff in diff_plots])

        vmin = max(-1.0, vmin)
        vmax = min(+1.0, vmax)

        for diff in diff_plots:

            assert len(diff.wavelength) == len(diff.diff)

            for offset in [-1, 0, 1]:
                ys = (diff.phase + offset) * np.ones(len(diff.wavelength))
                sc = plot.scatter(diff.wavelength, ys, s=1, c=diff.diff, cmap=args.cmap,
                                  vmin=min(vmin, -vmax), vmax=max(vmax, -vmin))

        fig.colorbar(sc)
        plt.show()

def _find_minimum(xs, ys, dx, range_AA, box_size_AA):

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    width = int(np.ceil(box_size_AA / dx))

    kernel = Box1DKernel(width)

    ys = convolve(ys, kernel=kernel, boundary=None)

    assert len(xs) == len(ys)

    # remove convolution boundaries
    clip = kernel.array.size // 2
    xs = xs[clip:-clip]
    ys = ys[clip:-clip]

    mask = [H_ALPHA.value - range_AA <= x <= H_ALPHA.value + range_AA for x in xs]
    xs = xs[mask]
    ys = ys[mask]

    i = np.argmin(ys)
    return xs[i]


if __name__ == '__main__':
    main()
