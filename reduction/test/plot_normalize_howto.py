"""
Plot an explanation how normalize works
"""

import os.path

import matplotlib.pyplot as plt
import numpy as np

from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.constants import H_ALPHA
from reduction.normalize import normalize
from reduction.plotting import setup_presentation_plots
from reduction.spectrum import Spectrum, find_minimum
from reduction.utils.ranges import closed_range


def main():

    setup_presentation_plots()

    padding_aa = 10.0
    box_aa = 0.5

    polynomial_degree = range(2, 10)

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

    color_spec = 'tab:blue'
    color_ref = 'tab:orange'
    color_diff = 'tab:green'
    color_fit = 'tab:red'
    color_norm = 'tab:purple'

    red_shift = min_spectrum - min_reference

    fig = plt.figure(figsize=(10, 6))
    fig.set_tight_layout(True)

    plot_221 = fig.add_subplot(221)
    plot_221.plot(spectrum.xs, spectrum.ys, label='meas', color=color_spec)
    plot_221.plot(spectrum.xs, reference_ys, label='synth', color=color_ref)
    plot_221.xaxis.set_major_formatter(plt.NullFormatter())
    plot_221.set_title('measured and reference spectrum')
    ylim = plot_221.get_ylim()

    reference_shifted_ys = AlgolHAlphaModel(redshift=red_shift, sigma=instrument_fwhm / 2.4)(spectrum.xs)

    plot_222 = fig.add_subplot(222)
    plot_222.plot(spectrum.xs, spectrum.ys, label='meas', color=color_spec)
    plot_222.plot(spectrum.xs, reference_ys, ':', label='synth', color=color_ref)
    plot_222.plot(spectrum.xs, reference_shifted_ys, label='synth with redshift and instrument FWHM',
                  color=color_ref)
    plot_222.xaxis.set_major_formatter(plt.NullFormatter())
    plot_222.yaxis.set_major_formatter(plt.NullFormatter())
    plot_222.set_ylim(ylim)
    plot_222.set_title('redshift and instrument FWHM')

    plot_224 = fig.add_subplot(224)

    normalization = normalize(spectrum.xs, spectrum.ys, reference_shifted_ys, polynomial_degree, continuum_ranges)

    plot_224.plot(spectrum.xs, spectrum.ys, label='meas', color=color_spec)
    plot_224.plot(spectrum.xs, spectrum.ys / reference_shifted_ys, label='$meas/synth$', color=color_diff)
    plot_224.plot(spectrum.xs, normalization.fit, label='best fit', color=color_fit)

    if continuum_ranges and continuum_ranges.is_bounded():
        for r in continuum_ranges.intervals():
            plot_224.axvspan(r[0], r[1], alpha=0.25)
            plot_224.text(0.5 * (r[0] + r[1]), 0.45, "fit range", ha="center", va="center", size=9)

    plot_224.yaxis.set_major_formatter(plt.NullFormatter())
    plot_224.set_ylim(ylim)
    plot_224.set_title('polynomial${}^{%d}$ = best_fit (meas / ref_ys)' % normalization.deg)

    plot_223 = fig.add_subplot(223)

    plot_223.plot(spectrum.xs, spectrum.ys, label='meas', color=color_spec)
    plot_223.plot(spectrum.xs, reference_shifted_ys, label='synth', color=color_ref)
    plot_223.plot(spectrum.xs, normalization.norm, label='normalized', color=color_norm)

    # if continuum_ranges and continuum_ranges.is_bounded():
    #     for r in continuum_ranges.intervals():
    #         plot_223.axvspan(r[0], r[1], alpha=0.25)
    #         plot_223.text(0.5 * (r[0] + r[1]), 0.45, "fit range", ha="center", va="center", size=9)

    plot_223.set_ylim(ylim)
    plot_223.set_title('normalized = meas / polynomial')

    from matplotlib.patches import ConnectionPatch

    p1 = ConnectionPatch(axesA=plot_221, xyA=(0.97, 0.2), coordsA='axes fraction',
                         axesB=plot_222, xyB=(0.03, 0.2), coordsB='axes fraction',
                         arrowstyle='simple', color='r', mutation_scale=30, figure=fig)

    p2 = ConnectionPatch(axesA=plot_222, xyA=(0.90, 0.03), coordsA='axes fraction',
                         axesB=plot_224, xyB=(0.90, 0.97), coordsB='axes fraction',
                         arrowstyle='simple', color='r', mutation_scale=30, figure=fig)

    p3 = ConnectionPatch(axesA=plot_224, xyA=(0.03, 0.2), coordsA='axes fraction',
                         axesB=plot_223, xyB=(0.97, 0.2), coordsB='axes fraction',
                         arrowstyle='simple', color='r', mutation_scale=30, figure=fig, zorder=3)

    fig.add_artist(p1)
    fig.add_artist(p2)
    fig.add_artist(p3)

    plt.show()


if __name__ == '__main__':
    main()
