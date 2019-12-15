#!python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.normalize import normalize_args, normalization_parser
from reduction.commandline import poly_iglob, filename_parser, verbose_parser, get_loglevel

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from os.path import basename

import numpy as np

import logging
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), normalization_parser(), verbose_parser],
                            fromfile_prefix_chars='@',
                            description='Display normalized spectrum using continuum ranges.')

    plot_parser = parser.add_mutually_exclusive_group()
    plot_parser.add_argument("--dont-plot", dest='plot', action='store_false', default=True,
                             help='do not display spectrum')

    parser.add_argument('--store-dat', metavar='filename.dat',
                        help='store object, reference and normalized spectrum as dat file')

    parser.add_argument('--store-fits', metavar='filename.fits',
                        help='store normalized spectrum as fits file. '
                             'For this to work, the original file must also be a fits file')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    for filename in poly_iglob(args.filenames):

        normalization = normalize_args(filename, args)

        if args.plot:
            plot = plt.axes()
            plot.set_title(basename(filename))
            normalization.plot(plot)
            plt.show()

        if args.store_dat:
            normalization.store_as_dat(args.store_dat)

        if args.store_fits:
            normalization.store_as_fits(filename, args.store_fits)

if __name__ == '__main__':
    main()
