#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display 1d fits data i.g. a spectrum via matplotlib
"""

from reduction.spectrum import load

import matplotlib.pyplot as plt
from astropy.io import fits

from argparse import ArgumentParser
import os.path

import logging
logger = logging.getLogger(__name__)


def show_file(filename, label, axes):

    x, y, unit = load(filename)

    axes.plot(x, y, label=label)
    axes.set_xlabel(unit or 'Wavelength')


def plot_many_files(args):

    fig = plt.figure()
    ax = None
    for i, filename in enumerate(sorted(args.filenames)):

        logger.info("display %s", filename)

        if args.merge:
            ax = ax or fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(len(args.filenames), 1, i + 1)

        if args.xrange:
            ax.set_xlim(args.xrange)

        label = os.path.basename(filename)
        show_file(filename, label, ax)

    plt.legend()


if __name__ == '__main__':
    parser = ArgumentParser(description='Plot spectrum arounf H_beta with continuum and integration ranges.')
    parser.add_argument('filenames', metavar='spectrum.fit', nargs='+',
                        help='one or more fit files containing H_beta')
    parser.add_argument('--xrange', nargs=2, type=float, metavar=('min', 'max'))
    parser.add_argument('--merge', default=False, action='store_true', help='show all spectra in a single plot')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--quiet', '-q', action='count', default=0)

    args = parser.parse_args()
    logging.basicConfig(level=max(0, logger.getEffectiveLevel() - 10 * args.verbose + 10 * args.quiet))

    plot_many_files(args)
    plt.show()
