#!python
# -*- coding: utf-8 -*-
"""
Display 2d fits data i.g. a spectrum via matplotlib
"""
import glob
import logging
import os
from argparse import ArgumentParser

import matplotlib
from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from reduction.commandline import filename_parser, get_loglevel, verbose_parser

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description='Plot one or more fits image files.',
                            parents=[filename_parser('image'), verbose_parser])
    parser.add_argument('--show-colorbar', dest='show_colorbar', default=False, action='store_true',
                        help="show image colorbar (default: False)")
    parser.add_argument('--dont-show-colorbar', dest='show_colorbar', default=False, action='store_false',
                        help="don't show image colorbar")

    args = parser.parse_args()
    logging.basicConfig(level=(get_loglevel(logger, args)))

    for filename in _multi_glob(args.filenames):

        with fits.open(filename) as hdu_list:
            images = [hdu.data for hdu in hdu_list if hdu.header.get("NAXIS", 0) == 2]
            if not images:
                logging.error(f"{filename} contains no images")

            for image in images:
                _display_image(filename, image, args.show_colorbar)


def _display_image(filename, image, show_colorbar):
    row_count, column_count = image.shape
    dpi = 300

    row_inches = (row_count + dpi - 1) // dpi
    column_inches = (column_count + dpi - 1) // dpi

    fig = plt.figure(figsize=(column_inches, row_inches), dpi=dpi)
    fig.canvas.manager.set_window_title(os.path.basename(filename))

    ax = fig.add_subplot()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(image, cmap=matplotlib.cm.gray)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    fig.tight_layout(pad=0)
    plt.show()
    plt.close(fig)


def _multi_glob(filenames):
    result = []
    for pattern in filenames:
        result.extend(glob.glob(pattern))

    return result


if __name__ == '__main__':
    main()
