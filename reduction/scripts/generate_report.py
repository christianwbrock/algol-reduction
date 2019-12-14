#!python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

import os
import os.path
from collections import namedtuple, defaultdict
from argparse import ArgumentParser
import logging

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams as plot_params

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.convolution import Box1DKernel
from astropy.convolution import convolve
from astropy.modeling.fitting import SimplexLSQFitter

from reduction.commandline import poly_glob, filename_parser, verbose_parser, get_loglevel
from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.spectrum import Spectrum
from reduction.stars.algol import Algol, algol_coordinate
from reduction.constants import H_ALPHA
from reduction.normalize import normalize
from reduction.utils.ranges import closed_range

from reduction.nan_statistics import nan_leastsquare


logger = logging.getLogger(__name__)

Diff = namedtuple('Diff', 'wavelength diff phase maxima')


def main():

    plot_params['figure.dpi'] = 150

    max_diff = 0.25
    padding = 10.0

    disc_range = closed_range(H_ALPHA.value - padding, H_ALPHA.value + padding)
    continuum_ranges = closed_range(6520, 6610) & ~disc_range

    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            description='Generate LaTeX report displaying spectrums normalized around the Halpha line.')

    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('-f', '--force', action='store_true')

    parser.add_argument('--fit-sigma', action='store_true',
                        help='Modify model stddev to fit best data')
    parser.add_argument('--fit-redshift', action='store_true',
                        help='Modify model redshift to best fit data')

    parser.add_argument('--deg', type=int, default=3,
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

    tex_file = open(os.path.join(args.output, "report.tex"), "w")

    tex_file.write("\\documentclass{article}\n")
    tex_file.write("\\usepackage[utf8]{inputenc}\n")
    tex_file.write("\\usepackage{graphicx}\n")
    tex_file.write("\\usepackage{seqsplit}\n")
    tex_file.write("\\usepackage{longtable}\n")
    tex_file.write("\\usepackage[hidelinks]{hyperref}\n")
    tex_file.write("\\title{Project Algol\\\\Spectrum reduction}\n")
    tex_file.write("\\date{\\today}\n")
    tex_file.write("\\author{%s\\\\by Christian Brock}\n" % os.path.basename(__file__).replace('_', '\\_'))
    tex_file.write("\\begin{document}\n")
    tex_file.write("\\maketitle\n")
    tex_file.write("\\begin{verbatim}\n")

    for k,v in args.__dict__.items():
        tex_file.write("--%s %s\n" % (k, v))

    tex_file.write("\\end{verbatim}\n")
    tex_file.write("\\tableofcontents\n")

    diff_image_name = "diff_by_phase.png"
    diff_image_wm_name = "diff_by_phase_with_maxima.png"
    sorted_diff_image_name = "diff_sorted_phase.png"
    sorted_diff_image_wm_name = "diff_sorted_phase_with_maxima.png"
    snr_by_observer_name = "snr_by_observer.png"

    tex_file.write("\n")
    tex_file.write("\\section{Final Result}\n")
    tex_file.write("\n")
    tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % diff_image_name)
    tex_file.write("\n")
    tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % diff_image_wm_name)
    tex_file.write("\n")
    tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % sorted_diff_image_name)
    tex_file.write("\n")
    tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % sorted_diff_image_wm_name)
    tex_file.write("\n")

    # list of Diffs
    diffs_by_phase = []
    snr_by_observer = defaultdict(list)

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
            tex_file.write("\section{%s}\n\n" % spectrum.observer)
            prev_observer = spectrum.observer
            prev_day = None

        obs_day = spectrum.obs_date.iso[:10]
        if obs_day != prev_day:
            tex_file.write("\subsection{%s}\n\n" % obs_day)
            prev_day = obs_day

        xs = spectrum.xs
        ys = spectrum.ys

        xs = xs[15:-15]
        ys = ys[15:-15]

        ys = ys / ys.max()

        obs_time = spectrum.obs_date
        res = spectrum.resolution

        # compute obs_time at solar system center
        light_travel_time = obs_time.light_travel_time(algol_coordinate, location=observer_location)
        obs_time += light_travel_time

        algol_rv_a = algol.rv_A(obs_time)
        radial_velocity_correction = algol_coordinate.radial_velocity_correction(obstime=obs_time,
                                                                                 location=observer_location)
        rv_predicted = algol_rv_a - radial_velocity_correction
        phase = algol.AB.phase(obs_time)

        def as_redshift(radial_velocity):
            return H_ALPHA * (radial_velocity / const.c).to(1)

        redshift_predicted = as_redshift(rv_predicted)
        redshift_from_data = u.Quantity(_find_minimum(xs, ys, spectrum.dx, 10, 1.5), u.AA) - H_ALPHA

        sigma = H_ALPHA / (res or 15000) / 2.354

        initial_model = AlgolHAlphaModel(redshift=redshift_from_data, sigma=sigma)
        initial_model.scale.fixed = True
        initial_model.redshift.fixed = not args.fit_redshift
        initial_model.sigma.fixed = not args.fit_sigma

        normalization = normalize(xs, ys, ref_ys=initial_model(xs), degree_or_range=args.deg, continuum_ranges=continuum_ranges)

        normalized = normalization.norm
        snr = normalization.snr
        normalization.plot(plt.figure().add_subplot(111))

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

            normalization = normalize(xs, ys, ref_ys=final_model(xs), degree_or_range=args.deg, continuum_ranges=continuum_ranges)

            normalized = normalization.norm
            snr = normalization.snr
            normalization.plot(plt.figure().add_subplot(111))

            image_norm2 = "%05d_norm2.png" % n
            plt.title("Normalization: %s" % final_model)
            plt.savefig(os.path.join(args.output, image_norm2))
            plt.close()
        else:
            image_norm2 = None
            final_model = initial_model

        image_diff = "%05d_diff.png" % n

        xlim = np.asarray(final_model.get_xlimits())
        xlim[0] = max(xlim[0], continuum_ranges.lower_bound())
        xlim[1] = min(xlim[1], continuum_ranges.upper_bound())

        diff_xs = xs - final_model.redshift
        diff_ys = normalized - final_model(xs)

        diff_mask = [x in disc_range for x in diff_xs]

        diff_xs = diff_xs[diff_mask]
        diff_ys = diff_ys[diff_mask]

        maxima = _find_maxima(diff_xs, diff_ys, H_ALPHA.value)

        diffs_by_phase.append(Diff(diff_xs, diff_ys, phase, maxima))
        if spectrum.resolution:
            snr_by_observer[spectrum.observer].append([spectrum.resolution, snr])

        create_diff_plot(final_model, initial_model, normalized, maxima, spectrum.short_name, xlim, xs, ys,
                         os.path.join(args.output, image_diff))

        def display(q, format_string):
            return ((format_string + " %s") % (q.value, q.unit)).replace('Angstrom', r'\AA')

        def display_rv(rv):
            return r"%.1f km/s, %.2f \AA" % (rv.to('km/s').value, as_redshift(rv).to('AA').value)

        tex_file.write("\n")
        tex_file.write("\\begin{center}\n")
        tex_file.write("\\begin{tabular}{|l|l|}\n")
        tex_file.write("\\hline\n")
        tex_file.write("Observer & %s \\\\\n" % spectrum.observer.replace('_', '\\_'))
        tex_file.write("Filename & \\seqsplit{%s} \\\\\n" % spectrum.short_name.replace('_', '\\_'))
        tex_file.write("\\hline\n")
        tex_file.write("Resolution $\\delta\\lambda/\\lambda$ & %s \\\\\n" % spectrum.resolution)
        tex_file.write("Sigma & %s \\\\\n" % display(sigma.to('AA'), "%.2f"))
        tex_file.write("SNR & %.0f \\\\\n" % snr)
        tex_file.write("\\hline\n")
        tex_file.write("Observation date $(UTC)$ & %s \\\\\n" % spectrum.obs_date.iso)
        tex_file.write("Light travel time& %s \\\\\n" % display(light_travel_time.to('min'), "%.1f"))
        tex_file.write("Phase & $%.2f$ \\\\\n" % phase)
        tex_file.write("\\hline\n")
        tex_file.write("Algol radial velocity & %s \\\\\n" % display_rv(algol_rv_a))
        tex_file.write("Barycentric correction & %s \\\\\n" % display_rv(radial_velocity_correction))
        tex_file.write("Final radial velocity& %s \\\\\n" % display_rv(rv_predicted))
        tex_file.write("\\hline\n")
        tex_file.write("Redshift, form data & %s \\\\\n" % display(redshift_from_data.to('AA'), "%.2f"))
        tex_file.write("\\hline\n")
        tex_file.write("\\end{tabular}\n")
        tex_file.write("\\end{center}\n")

        tex_file.write("\n")
        tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % image_diff)
        tex_file.write("\n")
        tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % image_norm1)
        tex_file.write("\n")
        if image_norm2:
            tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % image_norm2)
            tex_file.write("\n")
        tex_file.write("\\pagebreak\n")
        tex_file.write("\n")

    # end spectra

    diffs_by_phase = sorted(diffs_by_phase, key=lambda diff: diff.phase)

    vmin = max(-max_diff, np.min([np.nanmin(diff.diff) for diff in diffs_by_phase]))
    vmax = min(+max_diff, np.max([np.nanmax(diff.diff) for diff in diffs_by_phase]))

    plot_diff(args.cmap, args.output, diff_image_name, diffs_by_phase, disc_range, vmin, vmax, False)
    plot_diff(args.cmap, args.output, diff_image_wm_name, diffs_by_phase, disc_range, vmin, vmax, True)

    plot_sorted_diff(args.cmap, args.output, sorted_diff_image_name, diffs_by_phase, disc_range, vmin, vmax, False)
    plot_sorted_diff(args.cmap, args.output, sorted_diff_image_wm_name, diffs_by_phase, disc_range, vmin, vmax, True)

    plot_snr_by_observer(args.output, snr_by_observer_name, snr_by_observer)

    max_file = open(os.path.join(args.output, "maxima.dat"), "w")

    tex_file.write("\\appendix\n")
    tex_file.write("\\section{SNRs and Resolutions}\n")
    tex_file.write("\n")
    tex_file.write("\\includegraphics[width=\\textwidth]{%s}\n" % snr_by_observer_name)
    tex_file.write("\n")
    tex_file.write("\\section{maxima of differences}\n")
    tex_file.write("\n")
    tex_file.write("The raw date is stored in {\\tt %s}\n" % "maxima.dat")
    tex_file.write("\n")
    tex_file.write("\\begin{longtable}{|l|lll|lll|}\n")
    tex_file.write("\\hline\n")
    tex_file.write("phase & $\AA$ & $km/s$ & y & $\AA$ & $km/s$ & y \\\\\n")
    max_file.write("#phase,w1,v1,y1,w2,v2,y2\n")
    tex_file.write("\\hline\n")

    for diff in diffs_by_phase:

        if len(diff.maxima) == 2:
            x1, y1 = diff.maxima[0]
            x2, y2 = diff.maxima[1]
        elif len(diff.maxima) == 1:
            x, y = diff.maxima[0]
            if x < H_ALPHA.value:
                x1, y1 = x, y
                x2, y2 = None, None
            else:
                x1, y1 = None, None
                x2, y2 = x, y
        else:  # happens if both maxima are at the border
            continue

        v1 = ((x1 - H_ALPHA.value) / H_ALPHA.value * const.c).to('km/s').value if x1 else None
        v2 = ((x2 - H_ALPHA.value) / H_ALPHA.value * const.c).to('km/s').value if x2 else None

        def _(value, fmt):
            return fmt % value if value else ''

        tex_file.write("%.5f & %s & %s & %s  & %s & %s & %s\\\\\n" %
                       (diff.phase, _(x1, '%.1f'), _(v1, '%.0f'), _(y1, '%.3f'),
                        _(x2, '%.1f'), _(v2, '%.0f'), _(y2, '%.3f')))

        max_file.write("%.5f,%s,%s,%s,%s,%s,%s\n" %
                       (diff.phase, _(x1, '%.1f'), _(v1, '%.0f'), _(y1, '%.3f'),
                        _(x2, '%.1f'), _(v2, '%.0f'), _(y2, '%.3f')))

    tex_file.write("\\hline\n")
    tex_file.write("\\end{longtable}\n")
    tex_file.write("\n")
    tex_file.write("\n")
    tex_file.write("\\section{spectra by phase}\n")
    tex_file.write("\n")
    tex_file.write("\\begin{longtable}{|l|l|l|l|}\n")
    tex_file.write("\\hline\n")
    tex_file.write("phase & observer & date & filename \\\\\n")
    tex_file.write("\\hline\n")

    for spectrum in sorted(spectra, key=lambda sp: algol.AB.phase(sp.obs_date)):
        tex_file.write("%.5f & %s & %s & \\seqsplit{%s}\\\\\n" %
                       (algol.AB.phase(spectrum.obs_date), spectrum.observer.replace('_', '\\_'),
                        spectrum.obs_date.iso[:10], spectrum.short_name.replace('_', '\\_')))

    tex_file.write("\\hline\n")
    tex_file.write("\\end{longtable}\n")
    tex_file.write("\\end{document}\n")

    max_file.close()
    tex_file.close()


def plot_sorted_diff(args_cmap, args_output, sorted_diff_image_name, diffs_by_phase, disc_range, vmin, vmax, plot_maxima):
    # create the trailed spectrum *sorted* by phase plot
    fig = plt.figure(figsize=[6.4, 4.8 * 2])
    plot = fig.add_subplot(111)
    plot.set_xlim(disc_range.lower_bound(), disc_range.upper_bound())
    plot.set_ylabel('Spectra sorted by phase')
    plot.set_xlabel('Wavelength ($\AA$)')
    sc = None

    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []

    for i, diff in enumerate(diffs_by_phase):
        assert len(diff.wavelength) == len(diff.diff)

        ys = 1.0 * i * np.ones(len(diff.wavelength))
        sc = plot.scatter(diff.wavelength, ys, s=1, c=diff.diff, cmap=args_cmap, vmin=min(vmin, -vmax),
                          vmax=max(vmax, -vmin))

        if 0.15 <= diff.phase <= 0.85:
            for x, y in diff.maxima:
                if x < H_ALPHA.value:
                    left_xs.append(x)
                    left_ys.append(i)
                else:
                    right_xs.append(x)
                    right_ys.append(i)

    plot.vlines(H_ALPHA.value, *plot.get_ylim())

    if plot_maxima:
        plot.plot(left_xs, left_ys, 'k')
        plot.plot(right_xs, right_ys, 'k')

    ax2 = plot.twiny()
    ax2.set_xlim(((np.asarray(plot.get_xlim()) - H_ALPHA.value) / H_ALPHA.value * const.c).to('km/s').value)
    ax2.set_xlabel('Radial velocity ($km/s$)')

    fig.colorbar(sc)
    plt.savefig(os.path.join(args_output, sorted_diff_image_name))
    plt.close()


def plot_snr_by_observer(args_output, filename, snr_by_observer):

    assert isinstance(filename, str)
    assert isinstance(snr_by_observer, dict)

    fig = plt.figure()
    plot = fig.add_subplot(111)
    plot.set_xlabel('Resolution $\lambda / \delta \lambda$')
    plot.set_ylabel('SNR')

    for observer, resolutions_and_snrs in sorted(snr_by_observer.items()):
        resolutions = [i[0] for i in resolutions_and_snrs]
        snrs = [i[1] for i in resolutions_and_snrs]
        plot.scatter(resolutions, snrs, label=observer)

    plot.legend()
    plt.savefig(os.path.join(args_output, filename))
    plt.close(os.path.join(args_output, filename))


def plot_diff(args_cmap, args_output, diff_image_name, diffs_by_phase, disc_range, vmin, vmax, plot_maxima):
    """
    Create the trailed spectrum by phase plot
    """
    fig = plt.figure(figsize=[6.4, 4.8 * 2])
    plot = fig.add_subplot(111)
    plot.set_ylim(-0.5, 1.5)
    plot.set_xlim(disc_range.lower_bound(), disc_range.upper_bound())
    plot.set_ylabel('Phase')
    plot.set_xlabel('Wavelength ($\AA$)')
    for diff in diffs_by_phase:

        assert len(diff.wavelength) == len(diff.diff)

        for offset in [-1, 0, 1]:
            ys = (diff.phase + offset) * np.ones(len(diff.wavelength))
            sc = plot.scatter(diff.wavelength, ys, s=1, c=diff.diff, cmap=args_cmap, vmin=min(vmin, -vmax),
                              vmax=max(vmax, -vmin))
    plot.vlines(H_ALPHA.value, *plot.get_ylim())

    if plot_maxima:
        left_xs = []
        left_ys = []
        right_xs = []
        right_ys = []

        for diff in diffs_by_phase:

            if 0.15 <= diff.phase <= 0.85:
                for x, y in diff.maxima:
                    if x < H_ALPHA.value:
                        left_xs.append(x)
                        left_ys.append(diff.phase)
                    else:
                        right_xs.append(x)
                        right_ys.append(diff.phase)
        plot.plot(left_xs, left_ys, 'k')
        plot.plot(right_xs, right_ys, 'k')

    ax2 = plot.twiny()
    ax2.set_xlim(((np.asarray(plot.get_xlim()) - H_ALPHA.value) / H_ALPHA.value * const.c).to('km/s').value)
    ax2.set_xlabel('Radial velocity ($km/s$)')

    fig.colorbar(sc)

    plt.savefig(os.path.join(args_output, diff_image_name))
    plt.close()
    return sc


def create_diff_plot(final_model, initial_model, normalized, maxima, title, xlim, xs, ys, image_path):

    redshift = final_model.redshift

    plot = plt.figure().add_subplot(111)
    plot.set_ylim(-0.5, 1.5)
    plot.set_xlim(xlim)

    plot.plot(xs, 0.6 * ys, label='measured')
    plot.plot(xs, normalized, label='normalized')

    plot.plot(xs, initial_model(xs), label='predicted %s' % initial_model)
    if final_model is not initial_model:
        plot.plot(xs, final_model(xs), label='fitted %s' % final_model)

    plot.plot(xs, normalized - final_model(xs), label='normalized - fitted')

    if maxima:
        for x, y in maxima:
            plot.vlines(x + redshift, ymin=0, ymax=y, label='maxima')

    plot.hlines(0, xlim[0], xlim[1])
    plot.vlines(H_ALPHA.value + redshift, *plot.get_ylim())
    plot.set_title(title)
    plot.legend(loc='upper right')

    plt.savefig(image_path)
    plt.close()


def _find_maxima(xs, ys, center):

    result = []

    for r in [closed_range(np.min(xs), center), closed_range(center, np.max(xs))]:

        mask = [x in r for x in xs]

        ys_in_r = ys[mask]
        arg = np.argmax(ys_in_r)

        if arg == 0 or arg + 1 == len(ys_in_r):
            logger.debug('ignore maximum in %s at %s bound', r, 'lower' if arg == 0 else 'upper')
            continue

        y = ys_in_r[arg]
        x = xs[mask][arg]

        logger.debug('maximum in %s at x=%.1f, y=%.2f', r, x, y)
        result.append((x, y))

    return tuple(result)


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
