import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from reduction.stars.binary_orbit import BinaryOrbit


class Algol:
    """
    Algol orbital parameters are taken from interferometric results (see Baron2012).
    
    TODO: verify definition of period 
    """

    def __init__(self):

        self.coordinate = SkyCoord(47.04221855, 40.95564667, unit=u.deg, frame='icrs')

        self.rv = 3.7 * u.km / u.s
        # radial_velocity_error = 0 * kms

        self.distance_AB = 90 * u.lyr
        self.distance_AB_error = 3 * u.lyr

        # time_of_eclipse = Time(2441771.353, format='jd')
        time_of_eclipse = Time(2457403.333, format='jd')
        # time_of_eclipse = Time(2441773.49, format='jd')

        self.AB = BinaryOrbit(name1='AlgolA', name2='AlgolB',
                              # period=(2.8671362 * u.day),
                              period=(2.867328 * u.day),
                              epoch=time_of_eclipse,
                              m1=3.7 * u.solMass,
                              m2=0.81 * u.solMass,
                              e=0,
                              incl=98.6 * u.degree,
                              omega1=90 * u.degree,
                              Omega=47.4 * u.degree
                              )

        self.AB_C = BinaryOrbit(name1='AlgolAB', name2='AlgolC',
                                period=680.1 * u.day,
                                epoch=Time(2446936.4,  format='jd'),
                                m1=self.AB.M,
                                m2=1.6 * u.solMass,
                                e=0.227,
                                incl=83.76 * u.degree,
                                # omega1=310.8 * u.degree,
                                omega1=313.2 * u.degree,
                                Omega=132.7
                                )

        T_A = 12550 * u.K
        # T_B = 4900 * K
        # T_C = 7550 * K

    def phase_AB(self, time):
        return self.AB.phase(time)

    def phase_AB_C(self, time):
        return self.AB_C.phase(time)

    def rv_A(self, time):
        return self.rv + self.AB_C.v1(time) + self.AB.v1(time)

    def rv_B(self, time):
        return self.rv + self.AB_C.v1(time) + self.AB.v2(time)

    def rv_AB(self, time):
        return self.rv + self.AB_C.v1(time)

    def rv_C(self, time):
        return self.rv + self.AB_C.v2(time)
