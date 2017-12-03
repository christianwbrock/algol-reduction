#!python
# -*- coding: utf-8 -*-
"""
Assuming we do observations of a variable star defined by epoch and period
and store them in fits files having observer and date-obs header fields
we want to know what phases we have covered.
"""

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

import numpy as np

from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from reduction.stars.algol import algol_Kosmos as algol
from reduction.spectrum import load_obs_time


def get_obs_dates_by_observer(filenames):

    dates_by_observer = {}
    
    for filename in filenames:

        logger.debug("load file %s", filename)
        
        with fits.open(filename) as hdus:
            
            # print("%s\n" % hdus.info())
        
            assert len(hdus) > 0
            hdr = hdus[0].header
        
            obs = hdr.get('observer') or 'unknown'

            try:
                time, exposure = load_obs_time(hdus[0])
            except ValueError:
                logger.error("file '%s' contains no recognized observation time", filename)
                continue
            
            times = dates_by_observer.get(obs) or []
            times.append([time, exposure])
            
            dates_by_observer[obs] = times

    logger.debug("return %s", dates_by_observer)
    return dates_by_observer


def show_timeline(star, dates_by_observer, plot):
    
    for obs, times in sorted(dates_by_observer.items()):

        jds = [time[0].jd for time in times]
        phases = [star.phase_at(time[0]) for time in times]
        xerr = [(time[1] / star.period).to(1).value for time in times]
        yerr = np.full(len(times), fill_value=1)

        logger.debug("observer %s", obs)
        logger.debug("phases %s ...", phases[:5])
        logger.debug("jds %s ...", jds[:5])
        logger.debug("xerr %s ...", xerr[:5])
        logger.debug("yerr %s ...", yerr[:5])

        plot.errorbar(phases, jds, xerr=xerr, yerr=yerr, ls='none', elinewidth=2, label=obs)
    
    plot.legend()

    # use quarter-yearly y-axis
    locs=[Time(y, format='decimalyear').jd for y in np.arange(1900.0, 2100.0, 0.25)]
    plot.yaxis.set_major_locator(FixedLocator(locs))
    
    # show date, not the jd
    def jd_to_date(jd, pos=None):
        return Time(jd, format='jd').iso[0:10]

    plot.yaxis.set_major_formatter(FuncFormatter(jd_to_date))
    
    
if __name__ == '__main__':
    
    import sys
    
    filenames = sys.argv[1:]
    
    if len(filenames) == 0:
        print("%s spectrum.fits [second.fits ...]" % sys.argv[0])
        
    else:
        
        dates_by_observer = get_obs_dates_by_observer(filenames)


        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        show_timeline(algol, dates_by_observer, ax)
            
        plt.show()
