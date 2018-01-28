#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.commandline import poly_iglob, filename_parser, verbose_parser, set_loglevel
from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.spectrum import load_from_fit, load_obs_time
from reduction.stars.algol import Algol, algol_coordinate

from astropy.time import Time
from astropy import constants as const
from astropy import units as u

from argparse import ArgumentParser


from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            description='Display spectrum normalized around the Halpha line.',
                            epilog='An easy way to generate contimuum ranges is to use Richard O. Grays spectrum software and the extract_continuum ranges.py script.')
    args = parser.parse_args()
    set_loglevel(logger)

    algol = Algol()

    axes = plt.axes()
    for filename in poly_iglob(args.filenames):
        xs, ys, units = load_from_fit(filename)
        obs_time = load_obs_time(filename)

        obs_time -= obs_time.light_travel_time(algol_coordinate)
        rv = algol.rv_A(obs_time) - algol_coordinate.radial_velocity_correction(obstime=obs_time)

        dl = 6563. * u.AA * (rv / const.c).to(1)

        initial = AlgolHAlphaModel()


    axes.legend()
    plt.show()


if __name__ == '__main__':
    main()
