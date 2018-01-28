#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display 1d fits data i.g. a spectrum via matplotlib
"""

from reduction.spectrum import load
from reduction.commandline import poly_glob, filename_parser, get_loglevel, verbose_parser

import numpy as np

import matplotlib.pyplot as plt

from typing import List

from argparse import ArgumentParser
import os.path

import logging

logger = logging.getLogger(__name__)


def show_file(filename: str, normalize: bool, xrange: List[float], label: str, axes):
    x, y, unit = load(filename)

    if xrange:
        filter = [xrange[0] <= x <= xrange[1] for x in x]

        if np.any(filter):
            x = x[filter]
            y = y[filter]
        else:
            logger.error("ignore file %s where parameter xrange=%s is outsize data range=%s",
                         filename, xrange, [x[0], x[-1]])
            return

    if normalize:
        y /= max(y)

    axes.plot(x, y, label=label)
    axes.set_xlabel(unit or 'Wavelength')


def plot_many_files(args):
    fig = plt.figure()
    ax = None

    filenames = poly_glob(args.filenames)
    if not filenames:
        raise SystemExit("%s did not yield any filenames." % args.filenames)

    for i, filename in enumerate(sorted(filenames), start=1):

        logger.info("display %s", filename)

        if args.merge:
            ax = ax or fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(len(filenames), 1, i)

        if args.xrange:
            ax.set_xlim(args.xrange)

        label = os.path.basename(filename)
        show_file(filename, args.normalize, args.xrange, label, ax)

    plt.legend()


def main():
    parser = ArgumentParser(description='Plot one or more spectrum files.',
                            parents=[filename_parser('spectrum'), verbose_parser])
    parser.add_argument('--xrange', nargs=2, type=float, metavar=('min', 'max'), help='limit the plots x-range')
    parser.add_argument('--merge', '-m', default=False, action='store_true', help='show all spectra in a single plot')
    parser.add_argument('--normalize', '-n', default=False, action='store_true',
                        help='divide all spectra by their maximum value')

    args = parser.parse_args()

    logging.basicConfig(level=(get_loglevel(logger, args)))

    plot_many_files(args)
    plt.show()


if __name__ == '__main__':
    main()
