#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display 1d fits data i.g. a spectrum via matplotlib
"""

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from astropy.io import fits


def show_fits_1d(filename, title, plot):

    with fits.open(filename) as header_data_units:
        assert len(header_data_units) > 0

        logger.debug("%s", header_data_units[0]._summary())

        header = header_data_units[0].header
        data = header_data_units[0].data

        count = header['NAXIS1']
        x0 = header['CRVAL1']
        dx = header['CDELT1']

        assert header['NAXIS'] == 1
        assert dx > 0

        x = [x0 + i * dx for i in range(count)]
        y = data

        assert len(x) == len(y)

        plot.plot(x, y)
        plot.set_title(title)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    import sys
    
    num = len(sys.argv) - 1
    
    if num == 0:
        print("%s spectrum.fits [second.fits ...]" % sys.argv[0])
        
    else:
        # plt.switch_backend('Qt5Agg')
        
        fig = plt.figure()
        
        for i, filename in enumerate(sys.argv[1:]):
            logger.info("display %s", filename)
            ax = fig.add_subplot(num, 1, i+1)
            show_fits_1d(filename, filename, ax)
            
        plt.show()
