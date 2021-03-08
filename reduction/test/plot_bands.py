import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D
import numpy as np
import os.path

from collections import namedtuple
Band = namedtuple('Band', 'name mu fwhm color')


def main():
    
    stddev_to_fwhm = np.sqrt(8 * np.log(2))

    data = [
        Band(name='U', mu=365, fwhm=66, color='blueviolet'),
        Band(name='B', mu=445, fwhm=94, color='b'),
        Band(name='V', mu=551, fwhm=88, color='g'),
        #    Band(name='g', mu=464, fwhm=128, color='k'),
        Band(name='R', mu=658, fwhm=138, color='r'),
        Band(name='I', mu=806, fwhm=149, color='darkred'),
        Band(name='Z', mu=900, fwhm=None, color=None),
        Band(name='Y', mu=1020, fwhm=120, color=None),
        Band(name='J', mu=1220, fwhm=213, color=None),
        Band(name='H', mu=1630, fwhm=307, color=None),
        Band(name='K', mu=2190, fwhm=390, color=None),
        Band(name='L', mu=3450, fwhm=472, color=None),
        Band(name='M', mu=4750, fwhm=460, color=None),
        Band(name='N', mu=10500, fwhm=2500, color=None),
        Band(name='Q', mu=21001, fwhm=5800, color=None)
    ]
    fig = plt.figure()
    plot = fig.add_subplot(111)
    plot.set_title('Photometric spectrum')

    xmin = np.min([b.mu - b.fwhm for b in data if b.mu and b.fwhm])
    if False:
        xmax = 1000
    else:
        xmax = np.max([b.mu - b.fwhm for b in data if b.mu and b.fwhm])
        plot.set_xscale('log')

    xs = np.arange(xmin, xmax, 1)
    for band in data:

        if not (band.mu and band.fwhm):
            continue

        if not xmin <= band.mu <= xmax:
            continue

        g1 = Gaussian1D(mean=band.mu, stddev=band.fwhm / stddev_to_fwhm)

        plot.plot(xs, g1(xs), label=band.name, color=band.color)

    plot.axvspan(400, 700, alpha=0.2, color='#FFE87C')

    plot.legend()
    plot.set_xlabel('Wavelength ($nm$)')
    plt.show()


if __name__ == '__main__':
    main()
