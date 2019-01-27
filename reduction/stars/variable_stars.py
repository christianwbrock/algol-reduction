#!python
# -*- coding: utf-8 -*-
"""
Assuming we do observations of a variable star defined by epoch and period
and store them in fits files having observer and date-obs header fields
we want to know what phases we have covered.
"""

import logging
import math
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from astropy.time import Time

logger = logging.getLogger(__name__)


class VariableObject(object):
    """
    A periodic event can be converted between a time and phase.
    """

    def __init__(self, authority=None, coordinate=None):
        """
        Initialize an observable object having a phase per time function

        :param authority: who measured the parameters
        :param coordinate: SkyCoord of the object
        """

        self.authority = authority
        self.coordinate = coordinate

        if not coordinate:
            logger.error('Missing parameter coordinate required to compute radial_velocity_correction')
        else:
            assert isinstance(coordinate, SkyCoord)

    def to_1(self, time, location=None):
        raise NotImplemented

    def to_time(self, val, location=None):
        raise NotImplemented

    # noinspection PyMethodMayBeStatic
    def _light_travel_time(self, observation_time: Time, sky_coordinate: SkyCoord, observer_location=None):
        """"
        Due to the rotation around the sun, light from the target reaches earth sooner or later in a given six month
        period. Subclasses may be modify/disable this light_travel_time correction by overriding this method.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            observer_location = (observer_location or
                                 observation_time.location or
                                 EarthLocation.from_geocentric(0, 0, 0, u.meter))

        return observation_time.light_travel_time(sky_coordinate, kind='barycentric', location=observer_location)

    def phase_at(self, time, location=None):
        return self.to_1(time, location=location) % 1.0

    def period_at(self, time, location=None):
        return int(self.to_1(time, location=location))

    def next_after(self, time, next_, location=None):
        current = self.to_1(time, location=location)
        diff = (next_ - current) % 1.0
        assert 0 <= diff <= 1
        return self.to_time(current + diff, location=location)

    def previous_before(self, time, previous, location=None):
        current = self.to_1(time, location=location)
        diff = (previous - current) % 1.0
        assert 0 <= diff <= 1
        return self.to_time(current - previous, location=location)

    def plot_line(self, plot, color, t0, t1, y, location=None):
        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        plot.axhline(y=y, xmin=t0.jd, xmax=t1.jd, color=color, label=self.authority)

        first = self.period_at(t0, location)
        last = self.period_at(t1, location)

        for i in range(first, last + 1):
            x = self.to_time(i, location)

            plot.scatter(x.jd, y, color=color)

    def plot(self, plot, t0, t1, num=401):
        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        ts = Time(np.linspace(t0.jd, t1.jd, num=num), format='jd')

        def sin(t):
            return math.sin(self.to_1(t) * 2 * np.pi)

        x = [t.plot_date for t in ts]
        y = [sin(t) for t in ts]

        plot.plot(x, y, label=self.authority)


class RegularVariableObject(VariableObject):
    """
    Initialize an observable object having a phase per time function

    :param authority: who measured the parameters
    :param coordinate: SkyCoord of the object
    :param epoch: time when an maxima or minima was observed
    :param period: time range between maxima or minima
    :param assume_radial_velocity_correction: assume that epoch is radial velocite corrected
    """

    def __init__(self, epoch, period, authority=None, coordinate=None,
                 assume_radial_velocity_correction=True, location=None):
        super().__init__(authority, coordinate)

        if not isinstance(epoch, Time):
            epoch = Time(epoch)

        corr = 0 * u.second if assume_radial_velocity_correction else \
            self._light_travel_time(epoch, self.coordinate, observer_location=location)

        self.epoch = epoch - corr
        self.period = period

    def to_1(self, time, location=None):
        """
        Return the number of period since epoch.

        :param time: Observation time will be corrected
        :param location: Location of the observer, defaults to Greenwich
            For an earth bound observer passign a location makes no sense.
        :return: the number of period since epoch.
        """
        corr = self._light_travel_time(time, self.coordinate, observer_location=location)
        return ((time + corr - self.epoch) / self.period).to(1).value

    def to_time(self, val, location=None):
        """
        For a given number of periods since epoch.

        :param val:
        :param location:
        :return:
        """
        time = self.epoch + val * self.period
        corr = self._light_travel_time(time, self.coordinate, observer_location=location)
        return time - corr

    def __repr__(self):
        return "%s says that epoch: %s, period: %s, pos: %s" % (
            self.authority, self.epoch, self.period, self.coordinate)

