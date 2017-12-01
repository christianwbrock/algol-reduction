from unittest import TestCase

import matplotlib.pyplot as plt

from reduction.stars.mizar import Mizar


class TestMizar(TestCase):

    def test_print(self):

        mizar = Mizar()
        mizar.AB.plot_orbit(plt.axes(), mizar.rv)
        plt.show()
