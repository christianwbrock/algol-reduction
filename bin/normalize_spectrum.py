#!python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.normalize import normalize_args, arg_parser as normalization_parser
from reduction.commandline import poly_iglob, filename_parser, verbose_parser, get_loglevel

from argparse import ArgumentParser
from matplotlib import pyplot as plt

from os.path import basename

import logging
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), normalization_parser, verbose_parser],
                            description='Display normalized spectrum using continuum ranges.')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    for filename in poly_iglob(args.filenames):
        requested_plot = plt.axes()
        normalize_args(filename, args, requested_plot)
        requested_plot.set_title(basename(filename))
        plt.show()


if __name__ == '__main__':
    main()
