import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from reduction.stars.binary_orbit import BinaryOrbit
from reduction.stars.variable_stars import RegularVariableObject

import logging
logger = logging.getLogger(__name__)

algol_coordinate = SkyCoord(47.04221855, 40.95564667, unit=u.deg, frame='icrs')

unknown = RegularVariableObject(Time(2457403.333, format='jd'), 2.867328 * u.day, authority='unknown',
                                coordinate=algol_coordinate)
gcvs = RegularVariableObject(Time(2445641.5135, format='jd'), 2.8673043 * u.day, authority='GCVS',
                             coordinate=algol_coordinate)
kosmos_himmeljahr = RegularVariableObject(Time('2018-01-01T16:16:00', format='isot'), 2.8673043 * u.day,
                                          authority='Kosmos Himmelsjahr 2018', coordinate=algol_coordinate)
interstellarum = RegularVariableObject(Time('2018-01-18T20:34:00', format='isot'), 2.8673043 * u.day,
                                       authority='Himmels-Almanach 2018', coordinate=algol_coordinate)
baron2012 = RegularVariableObject(Time(2441771.353, format='jd'), 2.867328 * u.day,
                                  authority='Baron2012', coordinate=algol_coordinate)
zavala_2010 = RegularVariableObject(Time(2441773.49, format='jd'), 2.867328 * u.day,
                                    authority='Zavala2010', coordinate=algol_coordinate)
zavala2010_via_aaron2012 = RegularVariableObject(Time(2441771.3395, format='jd'), 2.867328 * u.day,
                                                 authority='Baron about Zavala?', coordinate=algol_coordinate)
aavso_my = RegularVariableObject(Time(2456181.84, format='jd'), 2.86736 * u.day,
                                 authority='AAVSO my calculation', coordinate=algol_coordinate)
aavso_self = RegularVariableObject(Time('2017-11-28T06:24', format='isot'), 2.86736 * u.day,
                                   authority='AAVSO their calculation', coordinate=algol_coordinate)
filipe_diaz = RegularVariableObject(Time('2017-08-14T03:40:00', format='isot'), 2.8673043 * u.day,
                                    authority='Filipe Diaz', coordinate=algol_coordinate,
                                    assume_radial_velocity_correction=False,
                                    location=EarthLocation(lat=+37.132*u.degree, lon=-8.365*u.degree))


class Algol:
    """
    TODO: verify definition of period 
    """

    def __init__(self):

        self.rv = 3.7 * u.km / u.s
        # radial_velocity_error = 0 * kms

        self.distance_AB = 90 * u.lyr
        self.distance_AB_error = 3 * u.lyr

        self.AB = BinaryOrbit(name1='AlgolA', name2='AlgolB',
                              # period=(2.8671362 * u.day),
                              period=unknown.period,
                              epoch=unknown.epoch,
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
