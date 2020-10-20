"""For a periodic star we want to schedule observations for a given time period.
"""

import logging
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime, timezone

import astropy.units as u
import numpy as np
from astroplan import Observer
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from icalendar import Calendar, Event

from reduction.commandline import time_parser, get_time_from_args, time_delta_parser, get_time_delta_from_args
from reduction.commandline import sky_coordinate_parser, get_sky_coord_from_args
from reduction.commandline import earth_location_parser, get_earth_location_from_args
from reduction.commandline import verbose_parser, get_loglevel
from reduction.stars.variable_stars import RegularVariableObject

logger = logging.getLogger(__name__)


def _create_schedule(filename, start_time, end_time, target_name, target, location, sun_horizon, star_horizon, phases,
                     epoch, period):
    observer = Observer(name="", location=location)

    logger.info("Time: %s -> %s", start_time, end_time)
    logger.info("Observer: %s", observer)
    logger.info("Star: %s", target)

    nights = _get_nights(observer, start_time, end_time, sun_horizon)

    time_above = _get_time_above_horizon(observer, start_time, end_time, target, star_horizon)

    obs_times = _intersection(nights, time_above)

    if epoch and period:
        variable_star = RegularVariableObject(epoch, period, coordinate=target, location=location)
    else:
        variable_star = None

    if phases and variable_star:
        phase_times = _get_phases(phases, start_time, end_time, variable_star)
        obs_times = _intersection(obs_times, phase_times)

    observations = _calculate_schedule(observer, obs_times, target, variable_star)

    _display_schedule(observations)

    _display_ical(filename, observer, observations, target_name)


@u.quantity_input(horizon=u.degree)
def _get_nights(observer, start_time, end_time, horizon):
    logger.info("Calculate nights with sun below %s" % horizon)

    if observer.is_night(start_time, horizon=horizon):
        dusk = start_time
    else:
        dusk = observer.sun_set_time(start_time, which='next', horizon=horizon)
        if dusk.mask:
            return [[]]

    dawn = observer.sun_rise_time(dusk, which='next', horizon=horizon)
    if dawn.mask:
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
def _get_time_above_horizon(observer, start_time, end_time, target_coordinate, horizon):
    logger.info("Calculate target time above %s" % horizon)

    target_is_up = observer.target_is_up(start_time, target_coordinate, horizon=horizon)
    if target_is_up:
        dusk = start_time
    else:
        dusk = observer.target_rise_time(start_time, target_coordinate, which='next', horizon=horizon)
        if dusk.mask:
            return [[]]

    dawn = observer.target_set_time(dusk, target_coordinate, which='next', horizon=horizon)
    if dawn.mask:
        return [[dusk, end_time]]

    nights = [[dusk, dawn]]

    while dawn < end_time:
        dusk = observer.target_rise_time(dawn, target_coordinate, which='next', horizon=horizon)
        dawn = observer.target_set_time(dusk, target_coordinate, which='next', horizon=horizon)

        if dusk.mask or dawn.mask:
            break

        nights.append([dusk, dawn])

    logger.info("Star rises %d times above %s between %s and %s", len(nights), horizon, start_time.iso, end_time.iso)
    logger.debug("at %s", nights)

    return nights


@u.quantity_input(time_range=u.day)
def _get_phases(phase, start_time, end_time, variable_star):
    logger.info("Calculate observation phases.")

    if not isinstance(start_time, Time):
        start_time = Time(start_time)

    if not isinstance(end_time, Time):
        end_time = Time(end_time)

    assert isinstance(phase, list)

    result = []

    if isinstance(phase[0], list):

        for ph in phase:
            result += _get_phases(ph, start_time, end_time, variable_star)

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


def _intersection(list0, list1):
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
    idx = {0, len(alt) - 1}
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


def _calculate_schedule(observer, obs_times, target, variable_star):
    result = []

    for i, [t0, t1] in enumerate(obs_times):

        assert isinstance(t0, Time)
        assert isinstance(t1, Time)

        comment = _altaz(observer, target, t0, t1)

        if variable_star:
            ph0 = variable_star.phase_at(t0, location=observer.location)
            ph1 = variable_star.phase_at(t1, location=observer.location)
        else:
            ph0 = ph1 = None

        result.append(Observation(i, ph0, ph1, t0, t1, comment))

    return result


def _display_schedule(observations):
    for obs in observations:
        assert isinstance(obs, Observation)
        if obs.ph0 and obs.ph1:
            logger.info("%2d: %.2f .. %.2f: %s .. %s   %s",
                        obs.nbr, obs.ph0, obs.ph1, obs.t0.iso[0:16], obs.t1.iso[11:16], obs.comment)
        else:
            logger.info("%2d: %s .. %s   %s",
                        obs.nbr, obs.t0.iso[0:16], obs.t1.iso[11:16], obs.comment)


@u.quantity_input(period=u.day)
def _display_ical(filename, observer, observations, target_name):
    logger.info('Write to \'%s\'', filename)

    cal = Calendar()
    cal['summary'] = target_name

    for obs in observations:
        event = Event()
        event.add('DTSTART', obs.t0.to_datetime(observer.timezone))
        event.add('DTEND', obs.t1.to_datetime(observer.timezone))
        comment = "%s Phase %.2f .. %.2f" % (target_name, obs.ph0, obs.ph1) if obs.ph0 and obs.ph1 else target_name
        event.add('SUMMARY', comment)
        event.add('DESCRIPTION', obs.comment)

        cal.add_component(event)

    with open(filename, 'wb') as file:
        file.write(cal.to_ical())


def main():
    # we use UTC as we have no observer location yet
    noon = Time(datetime.now(tz=timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0))

    parser = ArgumentParser(parents=[verbose_parser, time_parser('start', default=noon.isot),
                                     time_parser('end', default=(noon + 366 * u.day).isot),
                                     time_parser('epoch'),
                                     time_delta_parser('period'),
                                     sky_coordinate_parser(),
                                     earth_location_parser()],
                            fromfile_prefix_chars='@',
                            description='Plan observations for a possibly variable star.')

    parser.add_argument('-o', '--output', type=str, default='output.ics',
                        help='output file name (default: %(default)s)')
    parser.add_argument('--entry-prefix', type=str, metavar='str',
                        help='Begin of each calendar entry (default: target name or sky coordinates)')
    parser.add_argument('--phase', dest='phases', nargs=2, type=float, metavar=('min', 'max'), action='append',
                        help='Desired phase of the variable star. '
                             'If absent, all phases are shown. '
                             'This option can be used more than once.')
    parser.add_argument('--sun-below', metavar='deg', default=12, type=float,
                        help='Nights are whenever the sun is below the horizon further then this value '
                             '(default: %(default)s)')
    parser.add_argument('--star-above', metavar='deg', default=30, type=float,
                        help='Observation are scheduled when the star is above the horizon further then this value '
                             '(default: %(default)s)')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logging.INFO, args))

    start_time = get_time_from_args(args, 'start')
    end_time = get_time_from_args(args, 'end')

    phases = args.phases
    epoch = get_time_from_args(args, 'epoch', required=False)
    period = get_time_delta_from_args(args, 'period', required=bool(epoch))

    assert epoch and period or not epoch and not period

    target = get_sky_coord_from_args(args)

    star_name = args.entry_prefix or args.target_name or "ra=%s dec=%s" % (target.frame.ra, target.frame.dec)

    location = get_earth_location_from_args(args)

    sun_horizon = -1 * args.sun_below * u.degree
    star_horizon = +1 * args.star_above * u.degree
    filename = args.output

    _create_schedule(filename, start_time, end_time, star_name, target, location, sun_horizon, star_horizon, phases,
                     epoch, period)


if __name__ == '__main__':
    main()
