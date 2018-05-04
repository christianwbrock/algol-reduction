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

import numpy as np

import logging
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), normalization_parser, verbose_parser],
                            add_help=False, fromfile_prefix_chars='@',
                            description='Display normalized spectrum using continuum ranges.')

    plot_parser = parser.add_mutually_exclusive_group()
    plot_parser.add_argument("--dont-plot", dest='plot', action='store_false', default=True,
                             help='do not display spectrum')
    plot_parser.add_argument("--plot", dest='plot', action='store_true', help='display spectrum')

    parser.add_argument('--store-dat', metavar='filename.dat',
                        help='store object, reference and normalized spectrum as dat file')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    for filename in poly_iglob(args.filenames):
        if args.plot:
            requested_plot = plt.axes()
        else:
            requested_plot = None

        if args.store_dat:
            requested_spectra = {}
        else:
            requested_spectra = None

        normalize_args(filename, args, requested_plot, requested_spectra)

        if requested_plot:
            requested_plot.set_title(basename(filename))
            plt.show()

        if args.store_dat:
            xs = requested_spectra['xs']
            ys = requested_spectra['ys']
            ref_ys = requested_spectra['ref_ys']
            norm = requested_spectra['norm']

            if ref_ys is not None:
                data = [xs, ys, ref_ys, norm]
            else:
                data [xs, ys, norm]
            data = np.asarray(data)
            data = np.transpose(data)

            if ref_ys is not None:
                header = 'wavelength object reference normalized'
            else:
                header = 'wavelength object normalized'

            np.savetxt(args.store_dat, data, header=header)


if __name__ == '__main__':
    main()
