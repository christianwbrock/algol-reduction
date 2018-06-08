"""
Plot stuff having radial velocity, like orbits
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import astropy.constants as const
import astropy.units as u


def plot_rv_by_phase(plot, funcs, epoch, period, ref_wavelength=None, points=401,
                     show_xaxis=True, show_xlabel=True,
                     show_rv_axis=True, show_rv_label=True,
                     show_rs_axis=True, show_rs_label=True,
                     rv_label='radial velocity $(km/s)$',
                     rs_label='red-shift @$H\\alpha (\\AA)$'):
    """
    Plot radial velocity as phase plot displaying radial velocity and redshift axes.

    Parameters
    ----------

    plot: the plot to write to

    funcs: list of shape [n, 2] containing n functions to be plotted and there labels

    epoch: time of periastron, or ...

    period: of the epoch

    ref_wavelength: used to compute readshift from radial velocity

    point: number of points in plot

    show_xaxis: display the x-axis?
    """

    # we need additional redshift axis
    addy = plot.twinx()

    phase = np.linspace(0, 1, points)
    times = Time(np.linspace(epoch.jd, (epoch + period).jd, points), format='jd')

    for func, label in np.asarray(funcs):
        rv_kms = func(times).to('km/s')

        rs_min = _rf_to_rs(np.nanmin(rv_kms), ref_wavelength)
        rs_max = _rf_to_rs(np.nanmax(rv_kms), ref_wavelength)

        label = '%s $[%.2f .. %.2f \AA]$' % (label, rs_min.value, rs_max.value)

        plot.plot(phase, rv_kms.value, label=label)

    # assure both phase-scales match
    plot.set_xlim(-0.02, 1.02)

    rv_min, rv_max = plot.get_ylim() * u.km / u.s
    rs_min = _rf_to_rs (rv_min, ref_wavelength)
    rs_max = _rf_to_rs (rv_max, ref_wavelength)
    addy.set_ylim(rs_min.value, rs_max.value)

    if show_xaxis:
        if show_xlabel:
            plot.set_xlabel('Phase')
    else:
        plot.xaxis.set_major_locator(plt.NullLocator())

    if show_rv_axis:
        if show_rv_label:
            plot.set_ylabel(rv_label)
    else:
        plot.yaxis.set_major_locator(plt.NullLocator())

    if show_rs_axis:
        if show_rs_label:
            addy.set_ylabel(rs_label)
    else:
        addy.yaxis.set_major_locator(plt.NullLocator())

    plot.legend()


def _rf_to_rs(rv, ref_wavelength):
    return (rv / const.c).to(1) * ref_wavelength
