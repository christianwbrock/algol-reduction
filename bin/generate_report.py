#!/usr/bin/env python
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
from matplotlib import rcParams as plot_params

import numpy as np

import os
import os.path

from collections import namedtuple

import logging

logger = logging.getLogger(__name__)

Diff = namedtuple('Diff', 'wavelength diff phase')


def main():

    plot_params['figure.dpi'] = 150

    max_diff = 0.25
    padding = 10.0
    continuum_ranges = ((6520, H_ALPHA.value - padding), (H_ALPHA.value + padding, 6610))

    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            description='Generate LaTeX report displaying spectrums normalized around the Halpha line.')

    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('-f', '--force', action='store_true')

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

    os.makedirs(args.output, exist_ok=args.force)
    logger.info("write report to '%s'", os.path.abspath(args.output))

    if args.cmap not in cm.datad.keys():
        logger.warning('Invalid colormap not in %s', cm.datad.keys())
        args.cmap = parser.get_default('cmap')

    observer_location = EarthLocation.from_geodetic(lon=15.0, lat=50.0)
    algol = Algol()

    text_file = open(os.path.join(args.output, "report.tex"), "w")

    text_file.write("\\documentclass{article}\n")
    text_file.write("\\usepackage[utf8]{inputenc}\n")
    text_file.write("\\usepackage{graphicx}\n")
    text_file.write("\\usepackage{seqsplit}\n")
    text_file.write("\\usepackage[hidelinks]{hyperref}\n")
    text_file.write("\\title{Project Algol\\\\Spectrum reduction}\n")
    text_file.write("\\date{\\today}\n")
    text_file.write("\\author{%s\\\\by Christian Brock}\n" % os.path.basename(__file__).replace('_', '\\_'))
    text_file.write("\\begin{document}\n")
    text_file.write("\\maketitle\n")
    text_file.write("\\tableofcontents\n")

    diff_image_name = "diff_by_phase.png"
    sorted_diff_image_name = "diff_sorted_phase.png"
    text_file.write("\n")
    text_file.write("\\section{Final Result}\n")
    text_file.write("\n")
    text_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % diff_image_name)
    text_file.write("\n")
    text_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % sorted_diff_image_name)
    text_file.write("\n")

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

    prev_observer = None
    prev_day = None

    for n, spectrum in enumerate(sorted(spectra, key=lambda sp: (sp.observer, sp.obs_date)), start=1):

        logger.info("pass2 %d/%d: %s", n, len(spectra), spectrum.short_name)

        if spectrum.observer != prev_observer:
            text_file.write("\section{%s}\n\n" % spectrum.observer)
            prev_observer = spectrum.observer
            prev_day = None

        obs_day = spectrum.obs_date.iso[:10]
        if obs_day != prev_day:
            text_file.write("\subsection{%s}\n\n" % obs_day)
            prev_day = obs_day

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
        rv_predicted = algol_rv_a - radial_velocity_correction
        phase = algol.AB.phase(obs_time)

        def as_readshift(radial_velovity):
            return H_ALPHA * (radial_velovity / const.c).to(1)

        redshift_from_data = u.Quantity(_find_minimum(xs, ys, spectrum.dx, 10, 1.5), u.AA) - H_ALPHA

        sigma = H_ALPHA / (res or 15000)

        initial_model = AlgolHAlphaModel(redshift=redshift_from_data, sigma=sigma)
        initial_model.scale.fixed = True
        initial_model.redshift.fixed = not args.fit_redshift
        initial_model.sigma.fixed = not args.fit_sigma

        normalized, snr = normalize(xs, ys, ref_ys=initial_model(xs), deg=args.deg, continuum_ranges=continuum_ranges,
                               method=None, requested_plot=plt.figure().add_subplot(111))

        image_norm1 = "%05d_norm1.png" % n
        plt.title("Normalization: %s" % initial_model)
        plt.savefig(os.path.join(args.output, image_norm1))
        plt.close()

        if args.fit_sigma or args.fit_redshift:

            fitter = SimplexLSQFitter()
            fitter._stat_method = nan_leastsquare
            fitter._opt_method._maxiter = 2000

            final_model = fitter(initial_model, xs, normalized, weights=np.sqrt(normalized))

            logger.debug("fit info: %s", fitter.fit_info)

            normalized, snr = normalize(xs, ys, ref_ys=final_model(xs), deg=args.deg, continuum_ranges=continuum_ranges,
                                   method=None, requested_plot=plt.figure().add_subplot(111))

            image_norm2 = "%05d_norm2.png" % n
            plt.title("Normalization: %s" % final_model)
            plt.savefig(os.path.join(args.output, image_norm2))
            plt.close()
        else:
            image_norm2 = None
            final_model = initial_model

        image_diff = "%05d_diff.png" % n

        xlim = np.asarray(final_model.get_xlimits())
        xlim[0] = max(xlim[0], continuum_ranges[0][0])
        xlim[1] = min(xlim[1], continuum_ranges[-1][-1])

        create_diff_plot(final_model, initial_model, normalized, spectrum.short_name, xlim, xs, ys,
                         os.path.join(args.output, image_diff))

        def display(q, format_string):
            return ((format_string + " %s") % (q.value, q.unit)).replace('Angstrom', r'\AA')

        def display_rv(rv):
            return r"%.1f km/s, %.2f \AA" % (rv.to('km/s').value, as_readshift(rv).to('AA').value)

        text_file.write("\n")
        text_file.write("\\begin{center}\n")
        text_file.write("\\begin{tabular}{|l|l|}\n")
        text_file.write("\\hline\n")
        text_file.write("Observer & %s \\\\\n" % spectrum.observer.replace('_', '\\_'))
        text_file.write("Filename & \\seqsplit{%s} \\\\\n" % spectrum.short_name.replace('_', '\\_'))
        text_file.write("\\hline\n")
        text_file.write("Resolution $\\delta\\lambda/\\lambda$ & %s \\\\\n" % spectrum.resolution)
        text_file.write("Sigma & %s \\\\\n" % display(sigma.to('AA'), "%.2f"))
        text_file.write("SNR & %.0f \\\\\n" % snr)
        text_file.write("\\hline\n")
        text_file.write("Observation date $(UTC)$ & %s \\\\\n" % spectrum.obs_date.iso)
        text_file.write("Light travel time& %s \\\\\n" % display(light_travel_time.to('min'), "%.1f"))
        text_file.write("Phase & $%.2f$ \\\\\n" % phase)
        text_file.write("\\hline\n")
        text_file.write("Algol radial velocity & %s \\\\\n" % display_rv(algol_rv_a))
        text_file.write("Barycentric correction & %s \\\\\n" % display_rv(radial_velocity_correction))
        text_file.write("Final radial velocity& %s \\\\\n" % display_rv(rv_predicted))
        text_file.write("\\hline\n")
        text_file.write("Redshift, form data & %s \\\\\n" % display(redshift_from_data.to('AA'), "%.2f"))
        text_file.write("\\hline\n")
        text_file.write("\\end{tabular}\n")
        text_file.write("\\end{center}\n")

        text_file.write("\n")
        text_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % image_diff)
        text_file.write("\n")
        text_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % image_norm1)
        text_file.write("\n")
        if image_norm2:
            text_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % image_norm2)
            text_file.write("\n")
        text_file.write("\\pagebreak\n")
        text_file.write("\n")

        diff_plots.append(Diff(xs - final_model.redshift, normalized - final_model(xs), phase))

    # end spectra

    vmin = max(-max_diff, np.min([np.nanmin(diff.diff) for diff in diff_plots]))
    vmax = min(+max_diff, np.max([np.nanmax(diff.diff) for diff in diff_plots]))

    fig = plt.figure(figsize=[6.4, 4.8 * 2])
    plot = fig.add_subplot(111)
    plot.set_title("measured - predicted")

    plot.set_ylim(-0.5, 1.5)
    plot.set_xlim(H_ALPHA.value - padding, H_ALPHA.value + padding)
    plot.set_ylabel('Phase')

    for diff in diff_plots:

        assert len(diff.wavelength) == len(diff.diff)

        for offset in [-1, 0, 1]:
            ys = (diff.phase + offset) * np.ones(len(diff.wavelength))
            sc = plot.scatter(diff.wavelength, ys, s=1, c=diff.diff, cmap=args.cmap, vmin=min(vmin, -vmax),
                              vmax=max(vmax, -vmin))

    plot.vlines(H_ALPHA.value, *plot.get_ylim())

    fig.colorbar(sc)

    plt.savefig(os.path.join(args.output, diff_image_name))
    plt.close()

    fig = plt.figure(figsize=[6.4, 4.8 * 2])
    plot = fig.add_subplot(111)
    plot.set_title("measured - predicted")

    plot.set_xlim(H_ALPHA.value - padding, H_ALPHA.value + padding)
    plot.set_ylabel('Spectra sorted by phase')

    for i, diff in enumerate(sorted(diff_plots, key=lambda diff: diff.phase)):
        assert len(diff.wavelength) == len(diff.diff)

        ys = 1.0 * i * np.ones(len(diff.wavelength))
        sc = plot.scatter(diff.wavelength, ys, s=1, c=diff.diff, cmap=args.cmap, vmin=min(vmin, -vmax),
                          vmax=max(vmax, -vmin))

    plot.vlines(H_ALPHA.value, *plot.get_ylim())

    fig.colorbar(sc)

    plt.savefig(os.path.join(args.output, sorted_diff_image_name))
    plt.close()

    text_file.write("\\end{document}\n")


def create_diff_plot(final_model, initial_model, normalized, title, xlim, xs, ys, image_path):
    plot = plt.figure().add_subplot(111)
    plot.set_ylim(-0.5, 1.5)
    plot.set_xlim(xlim)
    plot.plot(xs, 0.6 * ys, label='measured')
    plot.plot(xs, normalized, label='normalized')
    plot.plot(xs, initial_model(xs), label='predicted %s' % initial_model)
    if final_model is not initial_model:
        plot.plot(xs, final_model(xs), label='fitted %s' % final_model)
    plot.plot(xs, normalized - final_model(xs), label='normalized - fitted')
    plot.hlines(0, xlim[0], xlim[1])
    plot.vlines(H_ALPHA.value, *plot.get_ylim())
    plot.set_title(title)
    plot.legend(loc='upper right')
    plt.savefig(image_path)
    plt.close()


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
