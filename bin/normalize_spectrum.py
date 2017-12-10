#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a single spectrum, display all x-ranges within [0.99 y-max .. ymax]
"""

from reduction.spectrum import load
from reduction.normalize import plot_normalized

from argparse import ArgumentParser
import os.path

from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser(description='Display normalized spectrum using continuum ranges.',
                            epilog='''An easy way to generate contimuum ranges is to use
                            Richard O. Grays spectrum software and the extract_continuum ranges.py script.''')
    parser.add_argument('filenames', nargs='+', metavar='spectrum.fit', help='one or more spectrum file')
    parser.add_argument('--degree', '-d', type=int, default=3)
    parser.add_argument('--continuum-range', '-c', dest='ranges', nargs=2, type=float, metavar=('xmin, xmax'),
                        action='append', required=True)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--quiet', '-q', action='count', default=0)

    args = parser.parse_args()
    logging.basicConfig(level=max(0, logger.getEffectiveLevel() - 10 * args.verbose + 10 * args.quiet))

    axes = plt.axes()

    for filename in args.filenames:

        plot_normalized(filename, args.ranges, [args.degree], axes)

    axes.legend()
    plt.show()
