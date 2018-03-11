import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from reduction.stars.binary_orbit import BinaryOrbit
from reduction.stars.variable_stars import RegularVariableObject


mizar_period_and_epoch = RegularVariableObject(Time(2447636.07, format='jd'), 20.53835 * u.day, "Ulrich Waldschlaeger",
                                               coordinate=SkyCoord("+13h23m55.5s +54d55m31s", frame='icrs'))


class Mizar:

    def __init__(self):

        self.rv = -6.31 * u.km / u.s
        # radial_velocity_error = 0 * kms

        self.distance_AB = 90 * u.lyr
        self.distance_AB_error = 3 * u.lyr

        self.AB = BinaryOrbit(period=20.53835 * u.day,
                              epoch=Time(2447636.07, format='jd'),
                              m1=2.43 * u.solMass,
                              m2=2.50 * u.solMass,
                              e=0.5354,
                              incl=60.5 * u.degree,
                              omega1=104.3 * u.degree,
                              Omega=0.5354 * u.degree
                              )

        # T_A = 12550 * K
        # T_B = 4900 * K

    def rv_A(self, time):
        return self.rv + self.AB.v1(time)

    def rv_B(self, time):
        return self.rv + self.AB.v2(time)


def plot_mizar():

    import matplotlib.pyplot as plt

    mizar = Mizar()
    mizar.AB.plot_orbit(plt.axes(), mizar.rv)
    plt.show()


if __name__ == '__main__':
    plot_mizar()

