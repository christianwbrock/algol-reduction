#!python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.normalize import normalize_args, arg_parser as normalization_parser
from reduction.commandline import poly_iglob, filename_parser, verbose_parser, get_loglevel

from reduction.spectrum import Spectrum

from argparse import ArgumentParser
from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), normalization_parser, verbose_parser],
                            description='Display normalized spectrum using continuum ranges.',
                            epilog='An easy way to generate reference spectra is to use Richard O. Grays spectrum'
                                   ' software.')
    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    for filename in poly_iglob(args.filenames):
        normalize_args(filename, args, requested_plot=plt.axes())
        plt.show()


if __name__ == '__main__':
    main()
