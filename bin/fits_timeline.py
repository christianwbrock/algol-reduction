#!python
# -*- coding: utf-8 -*-
"""
Assuming we do observations of a variable star defined by epoch and period
and store them in fits files having observer and date-obs header fields
we want to know what phases we have covered.
"""

from typing import *

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

from astropy.time import Time

from reduction.stars.algol import kosmos_himmeljahr as algol
from reduction.spectrum import Spectrum

from reduction.commandline import poly_iglob, filename_parser, verbose_parser, get_loglevel

from collections import defaultdict
from argparse import ArgumentParser

import logging
logger = logging.getLogger(__name__)


def get_obs_dates_by_observer(filenames: Iterable[str]):
    dates_by_observer = defaultdict(list)

    for filename in filenames:

        logger.debug("load file %s", filename)

        spectra = Spectrum.load_from_fit(filename, slice(None))

        for spectrum in spectra:

            if spectrum.obs_date is None:
                logger.error("file '%s' contains no recognized observation time", filename)
                continue

            obs = spectrum.observer or 'Unknown'

            dates_by_observer[obs].append([spectrum.obs_date, spectrum.exposure])

    logger.debug("return %s", dates_by_observer)
    return dates_by_observer


def show_time_line(star, dates_by_observer: Dict[str, List[Time]], plot: plt):
    for obs, times in sorted(dates_by_observer.items()):
        plot_dates = [time.plot_date for time, _ in times]
        phases = [star.phase_at(time) for time, _ in times]
        xerr = [(exposure / star.period).to(1).value / 2 for _, exposure in times]

        plot.errorbar(phases, plot_dates, xerr=xerr, yerr=None, ls='none', elinewidth=2, label=obs)

    locator = AutoDateLocator()
    plot.yaxis.set_major_locator(locator)
    plot.yaxis.set_major_formatter(AutoDateFormatter(locator))

    plt.gcf().autofmt_xdate()

    plt.grid()

    plot.legend()
    plot.set_xlabel('Phase')


def main():
    argument_parser = ArgumentParser(description='Show observation times grouped by observer.',
                                     parents=[filename_parser('spectrum'), verbose_parser])
    args = argument_parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    dates_by_observer = get_obs_dates_by_observer(poly_iglob(args.filenames))

    fig = plt.figure()

    axes = fig.add_subplot(1, 1, 1)
    axes.set_title('Observations by phase and date')
    show_time_line(algol, dates_by_observer, axes)

    plt.show()


if __name__ == '__main__':
    main()
