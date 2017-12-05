#!python
# -*- coding: utf-8 -*-
"""
Assuming we do observations of a variable star defined by epoch and period
and store them in fits files having observer and date-obs header fields
we want to know what phases we have covered.
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation

import astropy.units as u

import math


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

    def plot_line(self, plot, color, t0, t1, y, dy=0.1):
        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        plot.axhline(y=y, xmin=t0.jd, xmax=t1.jd, color=color, label=self.authority)

        first = self.period_at(t0)
        last = self.period_at(t1)

        for i in range(first, last + 1):
            x = self.to_time(i)

            plot.scatter(x.jd, y, color=color)

    def plot(self, plot, t0, t1, num=401, location=None):
        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        ts = Time(np.linspace(t0.jd, t1.jd, num=num), format='jd')

        def sin(t):
            return math.sin(self.to_1(t, location) * 2 * np.pi)

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

    @u.quantity_input(period=u.day)
    def __init__(self, epoch, period, authority=None, coordinate=None,
                 assume_radial_velocity_correction=True, location=None):
        super().__init__(authority, coordinate)

        if not isinstance(epoch, Time):
            epoch = Time(epoch)

        corr = 0 * u.second if assume_radial_velocity_correction else \
            epoch.light_travel_time(coordinate, location=location)

        self.epoch = epoch - corr
        self.period = period

    def to_1(self, time, location=None):
        corr = time.light_travel_time(self.coordinate, location=location)
        return ((time + corr - self.epoch) / self.period).to(1).value

    def to_time(self, val, location=None):
        time = self.epoch + val * self.period
        corr = time.light_travel_time(self.coordinate, location=location)
        return time - corr
