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
import astropy.units as u

import math


class VariableObject(object):
    """
    A periodic event can be converted between a time and phase.
    """

    def __init__(self, authority=None, coordinate=None):
        self.authority = authority
        self.coordinate = coordinate

    def to_1(self, time, observer=None):
        raise NotImplemented

    def to_time(self, val, observer=None):
        raise NotImplemented

    def phase_at(self, time, observer=None):
        return self.to_1(time, observer=observer) % 1.0

    def period_at(self, time, observer=None):
        return int(self.to_1(time, observer=observer))

    def next_after(self, time, next_, observer=None):
        current = self.to_1(time, observer=observer)
        diff = (next_ - current) % 1.0
        assert 0 <= diff <= 1
        return self.to_time(current + diff, observer=observer)

    def previous_before(self, time, previous, observer=None):
        current = self.to_1(time, observer=observer)
        diff = (previous - current) % 1.0
        assert 0 <= diff <= 1
        return self.to_time(current - previous, observer=observer)

    def plot_line(self, plot, color, t0, t1, y, dy=0.1):
        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        plot.axhline(y=y, xmin=t0.jd, xmax=t1.jd, color=color, label=self.authority)

        first = self.period_at(t0)
        last = self.period_at(t1)

        for i in range(first, last + 1):
            x = self.to_time(i)

            plot.scatter(x.jd, y, color=color)

    def plot(self, plot, t0, t1, num=401):
        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        ts = Time(np.linspace(t0.jd, t1.jd, num=num), format='jd')

        def sin(t):
            return math.sin(self.to_1(t) * 2 * np.pi)

        x = [t.jd for t in ts]
        y = [sin(t) for t in ts]

        plot.plot(x, y, label=self.authority)


class RegularVariableObject(VariableObject):
    
    @u.quantity_input(period=u.day)
    def __init__(self, epoch, period, authority=None, coordinate=None):
        super().__init__(authority, coordinate)

        if not isinstance(epoch, Time):
            epoch = Time(epoch)

        self.epoch = epoch
        self.period = period

    def to_1(self, time, observer=None):
        return ((time - self.epoch) / self.period).to(1).value

    def to_time(self, val, observer=None):
        return self.epoch + val * self.period
