"""
Plot the radial velocity of Algol AB and C components
"""

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
from astropy.time import Time

from reduction.constants import H_ALPHA
from reduction.plot_radial_velocity import plot_rv_by_phase
from reduction.plotting import setup_presentation_plots
from reduction.stars.algol import algol_coordinate

setup_presentation_plots()


def plot_algol_barycentric_correction():

    fig = plt.figure()
    fig.set_tight_layout(True)

    now = Time.now()
    one_year = 1 * u.year
    observer_location = EarthLocation.from_geodetic(lon=15.0, lat=50.0)

    def barycentric(t): return algol_coordinate.radial_velocity_correction(obstime=t, location=observer_location)

    plot = fig.add_subplot(1, 1, 1)
    plot.set_title('Earth toward Algol, duration=%.1f %s' % (one_year.value, one_year.unit))
    plot_rv_by_phase(plot, [[barycentric, 'barycentric correction']], now, one_year, H_ALPHA,
                     points=2001, rv_label='km/s', rs_label='$\\AA$')

    plt.show()


if __name__ == '__main__':
    plot_algol_barycentric_correction()
