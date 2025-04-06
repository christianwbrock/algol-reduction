"""
Plot the radial velocity of Algol AB and C components
"""

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
from astropy.time import Time

from reduction.constants import H_ALPHA
from reduction.plot_radial_velocity import plot_rv_by_phase
from reduction.plotting import setup_presentation_plots
from reduction.stars.algol import Algol, algol_coordinate

setup_presentation_plots()


def plot_algol_orbits():

    algol = Algol()

    fig = plt.figure()
    fig.set_tight_layout(True)

    plt_AB = fig.add_subplot(4, 1, 1)
    plt_AB.set_title('AlgolAB, duration=%.1f %s' % (algol.AB.period.value, algol.AB.period.unit))
    plot_rv_by_phase(plt_AB, [(algol.AB.v1, algol.AB.name1), (algol.AB.v2, algol.AB.name2)],
                     algol.AB.epoch, algol.AB.period, H_ALPHA,
                     show_xaxis=False, rv_label='km/s', rs_label='$\\AA$')

    plt_AB_C = fig.add_subplot(4, 1, 2)
    plt_AB_C.set_title('AlgolAB-C, duration=%.1f %s' % (algol.AB_C.period.value, algol.AB_C.period.unit))
    plot_rv_by_phase(plt_AB_C, [(algol.AB_C.v1, algol.AB_C.name1), (algol.AB_C.v2, algol.AB_C.name2)],
                     algol.AB_C.epoch, algol.AB_C.period, H_ALPHA,
                     show_xaxis=False, rv_label='km/s', rs_label='$\\AA$')

    now = Time.now()
    one_year = 1 * u.year
    observer_location = EarthLocation.from_geodetic(lon=15.0, lat=50.0)

    def barycentric(t): return algol_coordinate.radial_velocity_correction(obstime=t, location=observer_location)

    plt_bary = fig.add_subplot(4, 1, 3)
    plt_bary.set_title('Earth toward Algol, duration=%.1f %s' % (one_year.value, one_year.unit))
    plot_rv_by_phase(plt_bary, [[barycentric, 'barycentric correction']], now, one_year, H_ALPHA,
                     show_xaxis=False, points=2001, rv_label='km/s', rs_label='$\\AA$')

    def sum_a(t): return algol.rv + algol.AB.v1(t) + algol.AB_C.v1(t) - barycentric(t)
    sum_period = max(algol.AB.period, algol.AB_C.period, one_year)

    plt_sum = fig.add_subplot(4, 1, 4)
    plt_sum.set_title('sum above + $%.2f \\AA$, duration=%.1f %s' % (_rf_to_rs(algol.rv, H_ALPHA).value,
                                                                     sum_period.value, sum_period.unit))
    plot_rv_by_phase(plt_sum, [[sum_a, 'AlgolA']], now, sum_period, H_ALPHA,
                     show_xaxis=True, points=2001, rv_label='km/s', rs_label='$\\AA$')

    plt.show()


def _rf_to_rs(rv, ref_wavelength):
    return (rv / const.c).to(1) * ref_wavelength


if __name__ == '__main__':
    plot_algol_orbits()
