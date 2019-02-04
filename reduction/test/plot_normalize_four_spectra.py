"""
Display for Algol spectra in a figure
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np

from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.constants import H_ALPHA
from reduction.normalize import normalize, find_minimum
from reduction.plotting import setup_presentation_plots
from reduction.spectrum import Spectrum
from reduction.utils.ranges import closed_range

setup_presentation_plots()


filenames = [
    ('Algol2017/data/Periode_2017_2018/Uwe_Zurmuehl/2017_10_16/Algol_AT1B1Or_TOGi+Blk_171016_C1g2_ED80_434_72m60s_6104-6950lc-noh2o.fits',
     'Transmission grating'),
    ('Algol2017/data/Periode_2017_2018/Bernd/2017_10_19/Algol_2017_10_19_04-noh2o.fit', 'Lhires III'),
    ('Algol2017/data/Periode_2015_2016/Ulrich_Waldschläger_CT/algol_2015_10_10-noh2o_uw.fit', 'Czerny Turner'),
    ('Algol2017/data/Periode_2017_2018/Ulrich_Waldschläger_Echelle/Algol_180205_900s_uw_noh2o.fit', 'Echelle')
]

padding_aa = 10.0
box_aa = 0.5

polynomial_degree = range(2, 11)

all_range = closed_range(6400, 6700)
disc_range = closed_range(H_ALPHA.value - padding_aa, H_ALPHA.value + padding_aa)
h_alpha_range = closed_range(6520, 6610)
continuum_ranges = h_alpha_range & ~disc_range


def plot_spectra(range):
    fig = plt.figure()
    fig.set_tight_layout(True)

    for n, filename_and_spectrograph in enumerate(filenames, start=1):

        filename = filename_and_spectrograph[0]
        spectrograph = filename_and_spectrograph[1]

        filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', filename)

        spectrum = Spectrum.load(filename)

        xs = spectrum.xs[[x in range for x in spectrum.xs]]
        ys = spectrum.ys[[x in range for x in spectrum.xs]]

        ys /= np.nanmax(ys)

        plot = fig.add_subplot(2, 2, n)
        plot.plot(xs, ys)
        plot.set_title(spectrograph)
        plot.yaxis.set_major_formatter(plt.NullFormatter())

        if n <= 2:
            plot.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            plot.set_xlabel('Wavelength $(\\AA)$')

    plt.show()


def plot_normalization():

    fig = plt.figure()
    fig.set_tight_layout(True)

    for n, filename_and_spectrograph in enumerate(filenames, start=1):

        filename = filename_and_spectrograph[0]
        spectrograph = filename_and_spectrograph[1]

        filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', filename)

        spectrum = Spectrum.load(filename)

        xs = spectrum.xs[[x in h_alpha_range for x in spectrum.xs]]
        ys = spectrum.ys[[x in h_alpha_range for x in spectrum.xs]]

        ys /= np.nanmax(ys)
        ys *= 0.9

        redshift_from_data = find_minimum(spectrum, H_ALPHA.value - 10, H_ALPHA.value + 10, 1.5) - H_ALPHA.value
        sigma = H_ALPHA / (spectrum.resolution or 15000) / 2.354

        model = AlgolHAlphaModel(redshift=redshift_from_data, sigma=sigma)
        model.scale.fixed = True
        model.redshift.fixed = True
        model.sigma.fixed = True

        plot = fig.add_subplot(2, 2, n)

        normalization = normalize(xs, ys, ref_ys=model(xs), degree_or_range=polynomial_degree, continuum_ranges=continuum_ranges)

        normalization.plot(plot)
        plot.set_xlim(h_alpha_range.lower_bound() - 5, h_alpha_range.upper_bound() + 5)
        plot.set_ylim(0.4, 1.1)

        plot.legend_ = None
        # plot.set_title('%s; SNR=%.0f' % (spectrograph, snr))
        plot.set_title(spectrograph)
        if n in (2, 4):
            plot.yaxis.set_major_formatter(plt.NullFormatter())
        if n <= 2:
            plot.xaxis.set_major_formatter(plt.NullFormatter())

    plt.show()


def plot_diff():

    fig = plt.figure()
    # fig.set_tight_layout(True)
    fig.set_tight_layout(dict(w_pad=0.1, h_pad=0.1))

    for n, filename_and_spectrograph in enumerate(filenames, start=1):

        filename = filename_and_spectrograph[0]
        spectrograph = filename_and_spectrograph[1]

        filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', filename)

        spectrum = Spectrum.load(filename)

        xs = spectrum.xs[[x in h_alpha_range for x in spectrum.xs]]
        ys = spectrum.ys[[x in h_alpha_range for x in spectrum.xs]]

        ys /= np.nanmax(ys)

        redshift_from_data = find_minimum(spectrum, H_ALPHA.value - 10, H_ALPHA.value + 10, 1.5) - H_ALPHA.value
        sigma = H_ALPHA / (spectrum.resolution or 15000) / 2.354

        model = AlgolHAlphaModel(redshift=redshift_from_data, sigma=sigma)
        model.scale.fixed = True
        model.redshift.fixed = True
        model.sigma.fixed = True

        normalization = normalize(xs, ys, ref_ys=model(xs), degree_or_range=polynomial_degree, continuum_ranges=continuum_ranges)
        
        plot = fig.add_subplot(4, 2, n+0 if n <= 2 else n+2)

        plot.set_xlim(h_alpha_range.lower_bound() - 5, h_alpha_range.upper_bound() + 5)
        plot.set_ylim(0.4, 1.1)

        plot.plot(normalization.xs, normalization.ref_ys, color='tab:orange')
        plot.plot(normalization.xs, normalization.norm, color='tab:purple')

        plot.set_title(spectrograph)
        plot.xaxis.set_major_formatter(plt.NullFormatter())
        if n in (2, 4):
            plot.yaxis.set_major_formatter(plt.NullFormatter())

        plot = fig.add_subplot(4, 2, n+2 if n <= 2 else n+4)

        plot.plot(normalization.xs, normalization.norm - normalization.ref_ys, color='tab:green')

        plot.set_xlim(h_alpha_range.lower_bound() - 5, h_alpha_range.upper_bound() + 5)
        plot.set_ylim(-0.02, 0.17)

        # plot.set_title('%s; SNR=%.0f' % (spectrograph, snr))
        if n in (2, 4):
            plot.yaxis.set_major_formatter(plt.NullFormatter())
        if n in (1, 2):
            plot.xaxis.set_major_formatter(plt.NullFormatter())

    plt.show()


def plot_diff_error():

    fig = plt.figure()
    fig.set_tight_layout(dict(w_pad=0.1, h_pad=0.1))

    filename, spectrograph = filenames[0]
    filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', filename)

    spectrum = Spectrum.load(filename)

    xs = spectrum.xs[[x in h_alpha_range for x in spectrum.xs]]
    ys = spectrum.ys[[x in h_alpha_range for x in spectrum.xs]]

    ys /= np.nanmax(ys)

    redshift_from_data = find_minimum(spectrum, H_ALPHA.value - 10, H_ALPHA.value + 10, 1.5) - H_ALPHA.value
    sigma = H_ALPHA / (spectrum.resolution) / 2.354

    model = AlgolHAlphaModel(redshift=redshift_from_data, sigma=sigma)
    model.scale.fixed = True
    model.redshift.fixed = True
    model.sigma.fixed = True

    normalization_correct = normalize(xs, ys, ref_ys=model(xs), degree_or_range=polynomial_degree, continuum_ranges=continuum_ranges)

    for column, delta_rs in enumerate(np.linspace(-0.2, 0.2, 5)):
        for row, delta_resol in enumerate(np.linspace(-3000, 3000, 5)):

            sigma_error = H_ALPHA / (spectrum.resolution + delta_resol) / 2.354
            model_error = AlgolHAlphaModel(redshift=redshift_from_data + delta_rs, sigma=sigma_error)
            model_error.scale.fixed = True
            model_error.redshift.fixed = True
            model_error.sigma.fixed = True

            normalization_error = normalize(xs, ys, ref_ys=model_error(xs), degree_or_range=polynomial_degree, continuum_ranges=continuum_ranges)

            plot = fig.add_subplot(5, 5, 1 + 5 * row + column)

            if column==2 and row==2:
                plot.plot(xs, normalization_correct.norm - model(xs), '-', color='tab:green')
            else:
                plot.plot(xs, normalization_correct.norm - model(xs), ':', color='tab:green')
                plot.plot(xs, normalization_error.norm - model_error(xs), '-', color='tab:red', alpha=0.5)

            plot.set_ylim(-0.02, 0.17)
            plot.set_xlim(disc_range.points)

            plot.yaxis.set_major_formatter(plt.NullFormatter())
            plot.xaxis.set_major_formatter(plt.NullFormatter())

            if row == 0:
                plot.set_title('redshift\n$%+.1f \\AA$' % delta_rs)

            if column == 0:
                plot.set_ylabel('resol\n$%+2.0f$' % delta_resol)

    plt.show()


if __name__ == '__main__':

    plot_spectra(all_range)
    plot_spectra(h_alpha_range)
    plot_normalization()
    plot_diff()
    plot_diff_error()
