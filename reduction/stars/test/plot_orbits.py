#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the radial velocity of Algol AB and C components

TODO: plot heliocentric correction
"""

import matplotlib.pyplot as plt

from reduction.stars.algol import Algol, algol_coordinate

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.constants as const
import astropy.units as u

from reduction.constants import H_ALPHA

import numpy as np


def plot_heio_corr(plot):

    observer_location = EarthLocation.from_geodetic(lon=15.0, lat=50.0)
    t0 = Time.now()
    p = 365.25 * u.day

    times = Time(np.linspace(t0.jd, (t0 + p).jd, 401), format='jd')
    corr = [c.to('km/s').value for c in algol_coordinate.radial_velocity_correction(obstime=times,
                                                                                    location=observer_location)]

    plot.plot(times.jd, corr, label='heliocentric correction')

    # we need additional x and y axes
    addx = plot.twiny()
    addy = plot.twinx()

    # assure both x-scales match
    plot.set_xlim((t0 - 0.1 * p).jd, (t0 + 1.1 * p).jd)
    addx.set_xlim(-0.1, 1.1)

    # convert radial velocity to red-shift at H_alpha
    v_min, v_max = plot.get_ylim() * u.km / u.s
    l_min = (v_min / const.c).to(1) * H_ALPHA
    l_max = (v_max / const.c).to(1) * H_ALPHA
    addy.set_ylim(l_min.value, l_max.value)

    plot.xaxis.set_major_locator(plt.MaxNLocator(5))
    # plot.xaxis.set_major_locator(plt.LinearLocator())
    plot.xaxis.set_minor_locator(plt.MultipleLocator(1))
    plot.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    addx.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plot.set_ylabel('RV ($km/s$)')
    plot.set_xlabel('Julian date')
    addy.set_ylabel(r'$\delta\lambda at H\alpha (\AA)$')
    # addx.grid(True)
    plot.legend()


def plot_orbits_by_m2():

    algol = Algol()

    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(hspace=0.4)

    plt_AB = fig.add_subplot(3, 1, 1)
    algol.AB.plot_orbit(plt_AB)
    plt_AB.set_ylabel('')

    plt_AB_C = fig.add_subplot(3, 1, 2)
    algol.AB_C.plot_orbit(plt_AB_C)

    plt_helio = fig.add_subplot(3, 1, 3)
    plot_heio_corr(plt_helio)
    plt_helio.set_ylabel('')

    plt.show()


if __name__ == '__main__':
    plot_orbits_by_m2()
