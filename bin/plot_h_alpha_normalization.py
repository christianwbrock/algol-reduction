"""
Use this file to normalize spectra around H_alpha.
Several ranges are assumed to be in the continuum
"""

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from reduction.normalize import plot_normalized


def plot_normalization(filenames):

    degrees = [3, 5]
    ranges = [[6500, 6510], [6520, 6530], [6590, 6650]]

    range_min = ranges[0][0]
    range_max = ranges[-1][-1]
    range_size = range_max - range_min
    xlim = [range_min - 0.2 * range_size, range_max + 0.2 * range_size]
    ylim = [-0.1, 1.1]

    for i, filename in enumerate(filenames):
        ax = plt.subplot(len(filenames), 1, i + 1)

        logger.info("normalize %s", filename)
        plot_normalized(filename, ranges, degrees, ax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(filename)
        ax.legend()

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import sys
    if len(sys.argv) == 1:
        print('Usage: %s file1.fit [file2.fit ...]' % sys.argv[0])

    else:
        plot_normalization(sys.argv[1:])