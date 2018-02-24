# -*- coding: utf-8 -*-
"""
For a periodic star we want to schedule observations for a given time period.
"""

import sys
import os.path
from collections import namedtuple

import numpy as np

from astropy.time import Time
from astropy.coordinates import EarthLocation, Latitude, Longitude
import astropy.units as u

from astroplan import Observer, MAGIC_TIME

from icalendar import Calendar, Event

import logging
logger = logging.getLogger(__name__)


def create_shedule(filename=None):

    # from reduction.stars.delcep import delcep_GCVS as target
    # from reduction.stars.delcep import josi as target
    # target_name = 'Delta Cephei'

    from reduction.stars.algol import unknown as target
    target_name = 'Algol'

    # from reduction.stars.mizar import mizar as target
    # target_name = 'Mizar'

    # from reduction.observers import christian, v, filipe
    from reduction.observers import ulrich as location
    target_name += '_Marzahn'

    phases = None
    # phases = [0.0]
    # phases = [0.0, 0.2]

    start_time = Time.now()
    end_time = Time.now() + 366 * u.day

    # do not change below ---------------

    filename = filename or os.path.abspath(os.path.relpath(
        target_name.lower().replace(' ', '', -1) + '.ics'))

    observer = Observer(name="", location=location)

    logger.info("Time: %s -> %s", start_time, end_time)
    logger.info("Observer: %s", observer)
    logger.info("Star: %s", target)

    nights = get_nights(observer, start_time, end_time)

    time_above = get_time_above_horizon(observer, start_time, end_time, target.coordinate)

    obs_times = intersection(nights, time_above)

    if phases:
        phase_times = get_phases(phases, start_time, end_time, target)
        obs_times = intersection(obs_times, phase_times)

    observations = calculate_schedule(observer, obs_times, target)
    if filename:
        display_ical(filename, observer, observations, target_name)


@u.quantity_input(horizon=u.degree)
def get_nights(observer, start_time, end_time, horizon=-12*u.degree):

    logger.info("Calculate nights")
    
    if observer.is_night(start_time, horizon=horizon):
        dusk = start_time
    else:
        dusk = observer.sun_set_time(start_time, which='next', horizon=horizon)
        if dusk == MAGIC_TIME:
            return [[]]

    dawn = observer.sun_rise_time(dusk, which='next', horizon=horizon)
    if dawn == MAGIC_TIME:
        return [[dusk, end_time]]

    nights = [[dusk, dawn]]

    while dawn < end_time:
        dusk = observer.sun_set_time(dawn, which='next', horizon=horizon)
        dawn = observer.sun_rise_time(dusk, which='next', horizon=horizon)
    
        nights.append([dusk, dawn])

    logger.info("Sun sets %d times below %s between %s and %s", len(nights), horizon, start_time.iso, end_time.iso)
    logger.debug("at %s", nights)

    return nights


@u.quantity_input(time_range=u.day, horizon=u.degree)
def get_time_above_horizon(observer, start_time, end_time, target_coordinate, horizon=+30*u.degree):
    
    logger.info("Calculate target time above horizon.")

    target_is_up = observer.target_is_up(start_time, target_coordinate, horizon=horizon)
    if target_is_up:
        dusk = start_time
    else:
        dusk = observer.target_rise_time(start_time, target_coordinate, which='next', horizon=horizon)
        if dusk == MAGIC_TIME:
            return [[]]

    dawn = observer.target_set_time(dusk, target_coordinate, which='next', horizon=horizon)
    if dawn == MAGIC_TIME:
        return [[dusk, end_time]]

    nights = [[dusk, dawn]]

    while dawn < end_time:
        dusk = observer.target_rise_time(dawn, target_coordinate, which='next', horizon=horizon)
        dawn = observer.target_set_time(dusk, target_coordinate, which='next', horizon=horizon)
    
        if dusk == MAGIC_TIME or dawn == MAGIC_TIME:
            break

        nights.append([dusk, dawn])
    
    logger.info("Star rises %d times above %s between %s and %s", len(nights), horizon, start_time.iso, end_time.iso)
    logger.debug("at %s", nights)

    return nights


@u.quantity_input(time_range=u.day)
def get_phases(phase, start_time, end_time, variable_star):
    
    logger.info("Calculate observation phases.")

    if not isinstance(start_time, Time):
        start_time = Time(start_time)

    if not isinstance(end_time, Time):
        end_time = Time(end_time)
        
    assert isinstance(phase, list)
    
    result = []

    if isinstance(phase[0], list):
        
        for ph in phase:
            result += get_phases(ph, start_time, end_time, variable_star.epoch, variable_star.period)
            
    else:

        if isinstance(phase, (int, float)):
            phase_0 = phase_1 = phase

        elif len(phase) == 1:
            phase_0 = phase_1 = phase[0]

        else:
            assert len(phase) >= 2
            phase_0 = phase[0]
            phase_1 = phase[1]

            if not phase_0 <= phase_1:
                phase_0 -= 1.0

        assert phase_0 <= phase_1

        first = variable_star.period_at(start_time)
        last = variable_star.period_at(end_time)
        
        for i in range(first, last + 1):
            start = variable_star.to_time(i + phase_0)
            end = variable_star.to_time(i + phase_1)
            
            result.append([start, end])

    logger.info("There are %d phases between [%.2f .. %.2f] between %s and %s", len(result), phase_0, phase_1,
                start_time.iso, end_time.iso)
    logger.debug("at %s", result)

    return result


def _single_intersection(r0, r1):
    assert len(r0) == 2
    assert len(r1) == 2
    
    assert isinstance(r0[0], Time)
    assert isinstance(r0[1], Time)
    assert isinstance(r1[0], Time)
    assert isinstance(r1[1], Time)
    
    start = max(r0[0], r1[0])
    end = min(r0[1], r1[1])
    
    if start <= end:
        return [start, end]
    
    else:
        return None
    

def intersection(list0, list1):

    result = []
    for r0 in list0:
        for r1 in list1:
            inters = _single_intersection(r0, r1)
            if inters:
                result.append(inters)
    
    return result


def _az_string(az):
    """Return an azimuth angle as compass direction.
    
        >>> _az_string(0)
        'N'
        >>> _az_string(11)
        'N'
        >>> _az_string(12)
        'NNE'
        >>> _az_string(360 - 12)
        'NNW'
        >>> _az_string(360 - 11)
        'N'
    """
    assert 0.0 <= az <= 360

    compass = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    assert len(compass) == 16

    step = 360 / len(compass)
    
    idx = int(round(az / step)) % len(compass)
    assert 0 <= idx < len(compass)
    
    return compass[idx]


def _altaz(observer, coordinate, t0, t1, num=10):
    
    times = np.linspace(t0.jd, t1.jd, num=num)
    
    altaz_coord = [observer.altaz(Time(t, format='jd'), coordinate) for t in times]
    alt = np.asarray([coord.alt.to(u.deg).value for coord in altaz_coord])
    alt = np.asarray(np.round(alt), dtype=int)

    # determine first, last min and max position
    idx = {0, len(alt)-1}
    idx.add(np.argmin(alt))
    idx.add(np.argmax(alt))
    assert 2 <= len(idx) <= 4
    
    alt = alt[sorted(idx)]
    assert 2 <= len(alt) <= 4
    
    alt = [str(a) for a in alt]
    
    az0 = _az_string(altaz_coord[0].az.to(u.deg).value)
    az1 = _az_string(altaz_coord[1].az.to(u.deg).value)

    str_az = az0 if az0 == az1 else '->'.join([az0, az1])
    str_alt = alt[0] if len(alt) == 2 and alt[0] == alt[1] else '->'.join(alt)

    return "%s, %s deg" % (str_az, str_alt)


Observation = namedtuple('Observation', 'nbr ph0 ph1 t0 t1 comment')


def calculate_schedule(observer, obs_times, target):

    result = []

    for i, obs in enumerate(obs_times):
        t0 = obs[0]
        t1 = obs[1]

        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        ph0 = target.phase_at(t0, location=observer.location)
        ph1 = target.phase_at(t1, location=observer.location)

        comment = _altaz(observer, target.coordinate, t0, t1)

        logger.info("%2d: %.2f .. %.2f: %s .. %s   %s", i, ph0, ph1, t0.iso[0:16], t1.iso[11:16], comment)

        result.append(Observation(i, ph0, ph1, t0, t1, comment))

    return result


def display_schedule(observations):
    for obs in observations:
        assert isinstance(obs, Observation)
        logger.info("%2d: %.2f .. %.2f: %s .. %s   %s",
                    obs.nbr, obs.ph0, obs.ph1, obs.t0.iso[0:16], obs.t1.iso[11:16], obs.comment)
    

@u.quantity_input(period=u.day)
def display_ical(filename, observer, observations, target_name):

    logger.info('Write to \'%s\'', filename)
    
    cal = Calendar()
    cal['summary'] = target_name
    
    for obs in observations:
        event = Event()
        event.add('DTSTART', obs.t0.to_datetime(observer.timezone))
        event.add('DTEND', obs.t1.to_datetime(observer.timezone))
        event.add('SUMMARY', "%s Phase %.2f .. %.2f" % (target_name, obs.ph0, obs.ph1))
        event.add('DESCRIPTION', obs.comment)
        
        cal.add_component(event)
    
    with open(filename, 'wb') as file:
        file.write(cal.to_ical())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    create_shedule(filename)
