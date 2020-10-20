#!python
# -*- coding: utf-8 -*-
"""
Calculate heliocentric or barycentric correction for a star and a list of times.
"""
import logging
import sys
from argparse import ArgumentParser

from astropy.coordinates import SkyCoord
from astropy.time import Time

from reduction.commandline import earth_location_parser, get_earth_location_from_args
from reduction.commandline import sky_coordinate_parser, get_sky_coord_from_args
from reduction.commandline import verbose_parser, get_loglevel


def main():
    logger = logging.getLogger(__name__)

    parser = ArgumentParser(parents=[verbose_parser,
                                     sky_coordinate_parser(),
                                     earth_location_parser()],
                            description='''Run astropy.coordinates.SkyCoord.radial_velocity_correction().
                            Each line from stdin is interpreted as observation time and passed to astropy.time.Time.
                            
                            The result is written to stdout.
                            It has to be interpreted as velocity toward the observer (red-shift).
                            ''')
    parser.add_argument('--time-format', default='jd', help='Format of the time values in stdin')
    parser.add_argument('--result-unit', default='km/s', help='Unit of the results written to stdout')
    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logging.INFO, args))

    star = get_sky_coord_from_args(args)
    logger.info("star: %s", star)
    location = get_earth_location_from_args(args)
    logger.info("observer: %s", location.to_geodetic())

    for line in sys.stdin:
        if not line:  # ignore empty lines
            continue
        time = _make_time(line, format=args.time_format)
        correction = star.radial_velocity_correction(obstime=time, location=location)
        correction = correction.to(args.result_unit)
        sys.stdout.write("%f\n" % correction.value)


def _as_float(str):
    try:
        return float(str)
    except:
        return str


def _make_time(line, format='jd'):
    val = line.strip()
    return Time(val=_as_float(val), format=format)


if __name__ == '__main__':
    main()
