from unittest import TestCase

import os

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.time import Time

from reduction.observers import bernd

from reduction.constants import H_ALPHA
from reduction.stars.algol import Algol


class TestAlgol(TestCase):

    def test_phases_between_A_and_B(self):
        pass
        # algol = Algol()

        # self.assertAlmostEqual(algol.AB.phase(algol.AB.epoch), 0 * u.rad, delta=1*u.deg)
        # self.assertAlmostEqual(algol.AB.phase(algol.AB.epoch + 0.5 * algol.AB.period), np.pi * u.rad, delta=1*u.deg)
        # self.assertAlmostEqual(algol.AB.phase(algol.AB.epoch + 1.5 * algol.AB.period), np.pi * u.rad, delta=1*u.deg)
        #
        # self.assertAlmostEqual(algol.AB.phase(Time('2014-10-27T18:29:38')).to(u.rad).value, 0.1, delta=0.05)
        # self.assertAlmostEqual(algol.AB.phase(Time('2015-03-28T18:24:23')).to(u.rad).value, 0.8, delta=0.05)

    def test_print(self):

        import matplotlib.pyplot as plt

        algol = Algol()

        fig = plt.figure(figsize=(6,9))
        fig.subplots_adjust(hspace=0.6)

        # algol.AB.plot_orbit(fig.add_subplot(311), algol.rv)
        algol.AB_C.plot_orbit(fig.add_subplot(111), 0)

#        bernd.plot_heliocentric_correction(fig.add_subplot(413), Time('2017-06-01'), 30*u.day, algol.coordinate)
        # bernd.plot_heliocentric_correction(fig.add_subplot(313), Time('2017-01-01'), 365*u.day, algol.coordinate)

        plt.show()



