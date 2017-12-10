#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.spectrum import load
from argparse import ArgumentParser

import numpy as np

import logging
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser(description='Determine ranges where the spectrum is above y-limit.')
    parser.add_argument('filename', metavar='spectrum.fit', help='a spectrum file')
    parser.add_argument('--xrange', nargs=2, type=float, metavar=('min', 'max'),
                        help="restrict extraction to this part of the spectrum")
    parser.add_argument('--ylimit', nargs=1, type=float, metavar='min', default=0.99, help='y threshold')
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--normalize', dest='normalize', action='store_true', default=True,
                            help='normalize spectrum before applying y-limit')
    norm_group.add_argument('--dont-normalize', dest='normalize', action='store_false',
                            help='do not normalize spectrum before applying y-limit')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--quiet', '-q', action='count', default=0)

    args = parser.parse_args()
    logging.basicConfig(level=max(0, logger.getEffectiveLevel() - 10 * args.verbose + 10 * args.quiet))

    x, y, _ = load(args.filename)

    if args.normalize:
        y /= np.max(y)

    result = []

    prev = True
    prevStart = 0
    for i in range(1, len(x)):

        curr = y[i] > args.ylimit

        if prev and not curr:
            result.append([x[prevStart], x[i-1]])
        elif not prev and curr:
            prevStart = i

        prev = curr

    if prev:
        result.append([x[prevStart], x[-1]])

    print("ranges = %s" % result)

    print(" ".join(["-c %g %g" % (a, b) for a, b in result]))



