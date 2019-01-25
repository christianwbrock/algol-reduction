#!python
# -*- coding: utf-8 -*-
"""
Plot elevation and azimuth or a star for a given time range, e.g. an observation night.

You may have to run "pip install astroplan astropy" to install required libraries.
"""

from astroplan import Observer

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u

from matplotlib import pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

observer = Observer(name='Filipe?', location=EarthLocation(lon=-8.365, lat=+37.132, height=100.0))

# algol
star = SkyCoord(47.04221855, 40.95564667, unit=u.deg, frame='icrs')

# start and end time of observation
t0 = Time('2017-12-06 20:00')
t1 = Time('2017-12-07 08:00')


points = 200

ts = Time([t0 + (t1 - t0) * i / points for i in range(points + 1)])
aa = observer.altaz(ts, star)

axes = plt.axes()
axes.plot(ts.plot_date, aa.az.to(u.deg), label="azimuth (degree)")
axes.plot(ts.plot_date, aa.alt.to(u.deg), label="elevation (degree)")

axes.legend()
axes.grid()
locator = AutoDateLocator()
axes.xaxis.set_major_locator(locator)
axes.xaxis.set_major_formatter(AutoDateFormatter(locator))

plt.show()
