"""
Plot an explanation how normalize works
"""

import os.path

import numpy as np
import matplotlib.pyplot as plt

from reduction.spectrum import Spectrum, find_minimum
from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.normalize import normalize
from reduction.constants import H_ALPHA
from reduction.utils.ranges import closed_range


def main():

    padding_aa = 10.0
    box_aa = 0.5

    polynomial_degree = 7

    disc_range = closed_range(H_ALPHA.value - padding_aa, H_ALPHA.value + padding_aa)
    h_alpha_range = closed_range(6520, 6610)
    continuum_ranges = h_alpha_range & ~disc_range

    filename = 'Algol2017/data/Periode_2017_2018/Ulrich_Waldschläger_Echelle/Algol_180224_900s_uw-noh2o.fit'
    filename = os.path.join(os.path.dirname(__file__), '..', '..', '..', filename)

    spectrum = Spectrum.load(filename)
    instrument_fwhm = H_ALPHA.value / spectrum.resolution

    mask = [x in h_alpha_range for x in spectrum.xs]
    spectrum = Spectrum.from_arrays(spectrum.xs[mask], 0.9 * spectrum.ys[mask] / np.max(spectrum.ys[mask]))

    min_spectrum = find_minimum(spectrum, disc_range.lower_bound(), disc_range.upper_bound(), box_aa)

    reference_ys = AlgolHAlphaModel(redshift=0, sigma=0)(spectrum.xs)
    min_reference = find_minimum(Spectrum.from_arrays(spectrum.xs, reference_ys),
                                 disc_range.lower_bound(), disc_range.upper_bound(), box_aa)

    red_shift = min_spectrum - min_reference

    fig = plt.figure(figsize=(10, 6))
    fig.set_tight_layout(True)

    plot = fig.add_subplot(221)
    plot.plot(spectrum.xs, spectrum.ys, label='meas')
    plot.plot(spectrum.xs, reference_ys, label='synth')
    plot.xaxis.set_major_formatter(plt.NullFormatter())
    plot.set_title('measured and reference spectrum')
    ylim = plot.get_ylim()

    reference_shifted_ys = AlgolHAlphaModel(redshift=red_shift, sigma=instrument_fwhm / 2.4)(spectrum.xs)

    plot = fig.add_subplot(222)
    plot.plot(spectrum.xs, spectrum.ys, label='meas')
    lines = plot.plot(spectrum.xs, reference_ys, ':', label='synth')
    plot.plot(spectrum.xs, reference_shifted_ys, label='synth w/ redshift and instrument FWHM', color=lines[0].get_color())
    plot.xaxis.set_major_formatter(plt.NullFormatter())
    plot.yaxis.set_major_formatter(plt.NullFormatter())
    plot.set_ylim(ylim)
    plot.set_title('w/ redshift and instrument FWHM')

    plot = fig.add_subplot(223)

    requested_spectra = {}
    norm, snr = normalize(spectrum.xs, spectrum.ys, reference_shifted_ys, polynomial_degree, continuum_ranges,
                          method=None, requested_plot=None, requested_spectra=requested_spectra)

    plot.plot(spectrum.xs, spectrum.ys, label='meas')
    plot.plot(spectrum.xs, spectrum.ys / reference_shifted_ys, label='$meas/synth$')
    plot.plot(spectrum.xs, requested_spectra['fit'], label='best fit')

    if continuum_ranges and continuum_ranges.is_bounded():
        for r in continuum_ranges.intervals():
            plot.axvspan(r[0], r[1], alpha=0.25)

    plot.yaxis.set_major_formatter(plt.NullFormatter())
    plot.set_ylim(ylim)
    plot.set_title('best_fit of meas / ref (outside line)')

    plot = fig.add_subplot(224)

    plot.plot(spectrum.xs, spectrum.ys, label='meas',)
    plot.plot(spectrum.xs, reference_shifted_ys, label='synth')
    plot.plot(spectrum.xs, norm, label='normalized')

    if continuum_ranges and continuum_ranges.is_bounded():
        for r in continuum_ranges.intervals():
            plot.axvspan(r[0], r[1], alpha=0.25)

    plot.set_ylim(ylim)
    plot.set_title('normalized = meas / best_fit')

    plt.show()


if __name__ == '__main__':
    main()
