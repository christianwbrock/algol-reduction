#!python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.spectrum import Spectrum
from reduction.commandline import get_loglevel, verbose_parser

from argparse import ArgumentParser

import numpy as np

import logging
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Determine ranges where the spectrum is above y-limit.',
                            parents=[verbose_parser])
    parser.add_argument('filename', metavar='spectrum', help='a spectrum file')
    parser.add_argument('--index', type=int, default=0, help='hdu in the spectrum (default=0)')
    parser.add_argument('--xrange', nargs=2, type=float, metavar=('min', 'max'),
                        help="restrict extraction to this part of the spectrum")
    parser.add_argument('--ylimit', nargs=1, type=float, metavar='min', default=0.99, help='y threshold')
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--normalize', dest='normalize', action='store_true', default=True,
                            help='normalize spectrum before applying y-limit')
    norm_group.add_argument('--dont-normalize', dest='normalize', action='store_false',
                            help='do not normalize spectrum before applying y-limit')

    args = parser.parse_args()
    logging.basicConfig(level=get_loglevel(logger, args))

    spectrum = Spectrum.load(args.filename, slice(args.index, args.index+1))

    y = spectrum.ys
    x = spectrum.xs

    if args.normalize:
        logger.info("divide spectrum by %s", np.max(y))
        y /= np.max(y)

    if args.xrange:
        logger.info("limit wavelength range to %s", args.xrange)
        inside_range = [args.xrange[0] <= x <= args.xrange[1] for x in x]
        x = x[inside_range]
        y = y[inside_range]

        if not x:
            logger.error("%s does not overlap spectrum %s", args.xrange, args.filename)
            return

    y = y > args.ylimit

    result = []

    prev = y[0]
    prev_idx = 0
    for curr_idx in range(1, len(x)):

        curr = y[curr_idx]

        if prev and not curr:
            result.append([x[prev_idx], x[curr_idx - 1]])
        elif not prev and curr:
            prev_idx = curr_idx

        prev = curr

    if prev:
        result.append([x[prev_idx], x[-1]])

    print("ranges = %s" % result)

    print(" ".join(["-c %g %g" % (a, b) for a, b in result]))


if __name__ == '__main__':
    main()
