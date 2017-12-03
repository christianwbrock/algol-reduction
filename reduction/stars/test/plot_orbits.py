import matplotlib.pyplot as plt
import numpy as np

from reduction.stars.binary_orbit import BinaryOrbit

import astropy.units as u
from astropy.time import Time


def plot_orbits_by_m2():

    axes = plt.axes()

    for m2 in np.arange(1.5, 2.01, 0.1):
        orbit = BinaryOrbit(name1='AB', name2='C',
                            period=679.58 * u.day,
                            epoch=Time(2446931.6,  format='jd'),
                            m1=3.77 * u.solMass,
                            m2=m2 * u.solMass,
                            e=0.227,
                            incl=83.7 * u.degree,
                            # omega1=310.8 * u.degree,
                            omega1=80 * u.degree,
                            Omega=0)

        orbit.plot_orbit(axes, 0)

    axes.set_title('masses, I=83')
    plt.show()

if __name__ == '__main__':
    plot_orbits_by_m2()