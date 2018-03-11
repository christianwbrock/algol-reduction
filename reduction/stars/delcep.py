
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

import astropy.units as u

from reduction.stars.variable_stars import RegularVariableObject


# wikipedia
skyCoord = SkyCoord('22h29m10.26502s +58d24m54.7139s')


# delcep_Josi_orig = VariableStar("Josi initial period", Time(2457911, format='jd'), 5.366 * u.day)
josi = RegularVariableObject(Time(2457911.0875, format='jd'), 5.366341 * u.day, "Josi", skyCoord)
gcvs = RegularVariableObject(Time(2436075.445, format='jd'), 5.366341 * u.day, "GCVS", skyCoord)
aavso = RegularVariableObject(Time(2436075.445, format='jd'), 5.366266 * u.day, "AAVSO", skyCoord)
interstellarum_almanach = RegularVariableObject(Time('2018-01-01T14:00', format='isot'), 5.366266 * u.day,
                                                "Himmels Almanach 2012", skyCoord)
kosmos_himmeljahr = RegularVariableObject(Time('2018-01-01T20:00', format='isot'), 5.366266 * u.day,
                                          "Kosmos Himmelsjahr 2018", skyCoord)


def plot_comparison():

    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    axes = plt.axes()

    now = Time.now()
    than = now + 6 * u.day

    plt.axhline(y=0)

    for var in [josi, aavso, gcvs, interstellarum_almanach, kosmos_himmeljahr]:
        var.plot(axes, now, than)

    locator = AutoDateLocator()
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(AutoDateFormatter(locator))

    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_comparison()
