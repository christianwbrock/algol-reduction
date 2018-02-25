#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot the radial velocity of Algol AB and C components

TODO: plot heliocentric correction
"""

import matplotlib.pyplot as plt

from reduction.stars.algol import Algol


def plot_orbits_by_m2():

    algol = Algol()

    plt_AB = plt.subplot(2, 1, 1)
    algol.AB.plot_orbit(plt_AB)

    plt_AB_C = plt.subplot(2, 1, 2)
    algol.AB_C.plot_orbit(plt_AB_C)

    plt.show()


if __name__ == '__main__':
    plot_orbits_by_m2()
