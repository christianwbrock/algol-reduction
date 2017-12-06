from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from astropy import units as u

from reduction.observers import bernd
from reduction.stars.algol import algol_coordinate as star


def plot():

    earth_location = EarthLocation(lat=bernd.latitude, lon=bernd.longitude)

    t0 = Time.now()
    dur = 365 * u.day

    axes = plt.axes()

    bernd.plot_heliocentric_correction(axes, t0, dur, star, points=7000)

    ts = Time([t0 + i * dur/7000 for i in range(7000)])

    time_correction = ts.light_travel_time(star, location=earth_location)
    axes.plot(ts.plot_date, time_correction.to(u.min).value, label='time_correction')

    heliocentric_corr = star.radial_velocity_correction(obstime=ts, location=earth_location)
    axes.plot(ts.plot_date, heliocentric_corr.to(u.km/u.second).value, label='SkyCoord.radial_velocity_correction')

    locator = AutoDateLocator()
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(AutoDateFormatter(locator))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
