

import numpy as np
import matplotlib.pyplot as plt


from astropy.time import Time
import astropy.constants as const
import astropy.units as u

from astropy.coordinates.solar_system import get_body_barycentric_posvel
from astropy.coordinates.representation import UnitSphericalRepresentation
from astropy.coordinates.builtin_frames import GCRS
from astropy.coordinates import EarthLocation

from reduction.constants import H_ALPHA


class Observer:

    def __init__(self, name, latitude, longitude, height=None):

        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.height = height

    def _helio_vector(self, t, loc):
        """
        Compute the heliocentric velocity correction at a given time and place.
        Paramters
        ---------
        t : astropy.time.Time
            Time of the observation. Can be a Time array.
        loc : astropy.coordinates.EarthLocation
            The observer location at which to compute the correction.
        """
        vsun = get_body_barycentric_posvel('sun', t)[1]
        vearth = get_body_barycentric_posvel('earth', t)[1]

        vsunearth = vearth  # - vsun

        _, gcrs_v = loc.get_gcrs_posvel(t)

        return vsunearth + gcrs_v

    def _helio_corr(self, t, loc, target):
        """
        Compute the correction required to convert a radial velocity at a given
        time and place to a heliocentric velocity.
        Paramters
        ---------
        t : astropy.time.Time
            Time of the observation. Can be a Time array.
        loc : astropy.coordinates.EarthLocation
            The observer location at which to compute the correction.
        target : astropy.coordinates.SkyCoord
            The on-sky location at which to compute the correction.
        Returns
        -------
        vcorr : astropy.units.Quantity with velocity units
            The heliocentric correction with a positive sign.  I.e., *add* this
            to an observed radial velocity to get the heliocentric velocity.
        """
        vsuntarg_cartrepr = self._helio_vector(t, loc)
        gcrs_p, gcrs_v = loc.get_gcrs_posvel(t)

        gtarg = target.transform_to(GCRS(obstime=t, obsgeoloc=gcrs_p, obsgeovel=gcrs_v))
        targcart = gtarg.represent_as(UnitSphericalRepresentation).to_cartesian()
        res = targcart.dot(vsuntarg_cartrepr).to(u.km/u.s)
        return res


    def heliocentric_correction(self, t, sky_coord):

        loc = EarthLocation.from_geodetic(lat=self.latitude, lon=self.longitude, height=self.height)

        if isinstance(t, np.ndarray):
            return np.array([self._helio_corr(ti, loc, sky_coord) for ti in t])

        return self._helio_corr(t, loc, sky_coord)


    def heliocentric_correction_pa(self, t, sky_coord):
        """Use py astronomy to calculate heliocentric correction.

        :param t: astropy Time value or array 
        :param sky_coord: astropy SkyCoord
        
        :return: the radial velocity of the observer against the point in heaven 
        """

        import PyAstronomy.pyasl

        def corr(t0):
            v_kms, hjd = PyAstronomy.pyasl.helcorr(self.longitude, self.latitude, self.height, sky_coord.ra, sky_coord.dec,
                                                t0)
            return -v_kms

        if isinstance(t, Time):
            t = t.jd

        if isinstance(t, np.ndarray):
            res = np.array([corr(ti) for ti in t])
        else:
            res = corr(t)

        return res * u.km / u.s

    def plot_heliocentric_correction(self, plot, t0, p, sky_coord, v0=None, points=201):
        """Plot a time range via matplotlib 

        :param t0: plot start time 
        :param p: plot time range
        :param sky_coord: astropy SkyCoord
        :param v0: system radial velocity 
        :param points: number of points in x directions
        """

        t = Time(np.linspace(t0.jd, (t0 + p).jd, points), format='jd')

        # we need additional x and y axes
        addx = plot.twiny()
        addy = plot.twinx()

        v1 = self.heliocentric_correction(t, sky_coord)
        plot.plot(t.jd, v1, label=self.name)

        if v0:
            v_0 = np.ones(t.size) * v0
            plot.plot(t.jd, v_0, label=r'%.0f %s' % (v0.to('km/s').value, v0.to('km/s').unit))

        # assure both x-scales match
        plot.set_xlim((t0 - 0.1 * p).jd, (t0 + 1.1 * p).jd)
        addx.set_xlim(- (0.1 * p).to(u.day).value, (1.1 * p).to(u.day).value)

        # convert radial velocity to red-shift at H_alpha
        v_min, v_max = plot.get_ylim() * u.km / u.s
        l_min = (v_min / const.c).to(1) * H_ALPHA
        l_max = (v_max / const.c).to(1) * H_ALPHA
        addy.set_ylim(l_min.value, l_max.value)

        plot.xaxis.set_major_locator(plt.MaxNLocator(5))
        # plot.xaxis.set_major_locator(plt.MultipleLocator(10))
        plot.xaxis.set_minor_locator(plt.MultipleLocator(1))
        plot.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

        # addx.xaxis.set_minor_locator(plt.MultipleLocator(0.1))


        plot.set_ylabel('Radial velocity (km/s)')
        plot.set_xlabel('Julian date')
        addx.set_xlabel('Days')
        addy.set_ylabel(r'$\delta\lambda \| H\alpha (\AA)$')
        addx.grid(True)
        plot.legend()

bernd = Observer("Bernd", 51.003557, 13.708425, 270)
christian = Observer(r"Gönnsdorf", 51, 13.7, 300)