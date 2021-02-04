# play around w/ Voigt profiles

import logging
from argparse import ArgumentParser

import astropy.constants as const
import matplotlib.pyplot as plt
from astropy.time import Time

from reduction.commandline import earth_location_parser, get_earth_location_from_args
from reduction.commandline import sky_coordinate_parser, get_sky_coord_from_args
from reduction.commandline import time_delta_parser, get_time_delta_from_args
from reduction.commandline import time_parser, get_time_from_args
from reduction.commandline import verbose_parser, get_loglevel
from reduction.stars.variable_stars import RegularVariableObject

logger = logging.getLogger(__name__)


def main():
    args = _arg_parser().parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    csv_file = open(args.store_csv, 'w') if args.store_csv else None
    sky_coord = get_sky_coord_from_args(args)
    location = get_earth_location_from_args(args)
    reference_wavelength = float(args.reference_wavelength)
    epoch = get_time_from_args(args, 'epoch')
    period = get_time_delta_from_args(args, 'period')
    star = RegularVariableObject(epoch, period, location=location, coordinate=sky_coord)

    times = []
    wavelength = []
    absorption = []
    errors = []

    with open(args.csv_file) as f:
        for line in f:
            _, a, e, w, t = line.split()
            times.append(Time(t))
            wavelength.append(float(w))
            absorption.append(float(a))
            errors.append(float(e))

    def correct(wavelength, time):
        delta_lambda = (wavelength - reference_wavelength)
        delta_v_h_alpha = delta_lambda / reference_wavelength * const.c
        radial_velocity_correction = sky_coord.radial_velocity_correction(obstime=time, location=location)

        delta_v_kms = (delta_v_h_alpha + radial_velocity_correction).to('km/s').value
        return delta_v_kms

    delta_v = [correct(wavelength[i], times[i]) for i in range(len(times))]
    phases = [star.phase_at(time) for time in times]

    absorption = _sort_by_time(absorption, phases)
    errors = _sort_by_time(errors, phases)
    delta_v = _sort_by_time(delta_v, phases)
    wavelength = _sort_by_time(wavelength, phases)
    times = _sort_by_time(times, phases)
    phases = sorted(phases)

    if args.plot:
        # plt.title('Radial velocity measured by the shift of the $H_\\alpha$ line')
        plt.scatter(_three_phases(phases), _three_data(delta_v))
        plt.xlim(-0.5, 1.5)
        plt.ylabel('radial velocity (km/s)')
        plt.xlabel('phase')
        plt.show()
        plt.cla()

        # plt.title('$H_\\alpha$ absorption measured by fitting the line w/ voigt profiles')
        plt.errorbar(_three_phases(phases), _three_data(absorption), yerr=_three_data(errors), fmt="o")
        plt.xlim(-0.5, 1.5)
        plt.ylabel('$H_\\alpha absorption (\\AA)$')
        plt.xlabel('phase')
        plt.show()

    if csv_file:
        for i in range(len(times)):
            csv_file.write('%s\t%s\t%s\t%s\t%s\t%s\n' %
                           (times[i], times[i], absorption[i], errors[i], delta_v[i], wavelength[i]))

        csv_file.close()


def _arg_parser():
    parser = ArgumentParser(parents=[verbose_parser, sky_coordinate_parser(), earth_location_parser(),
                                     time_parser('epoch'), time_delta_parser('period')],
                            fromfile_prefix_chars='@',
                            description="""\
                            Plot results written by fit_three_voigts.
                            A barycentrically corrected radial velocity is computed from the reference wavelength. 
                            """)
    parser.add_argument("csv_file", help='Result file written by fit_three_voigts')
    parser.add_argument("--reference-wavelength", default=6562.8,
                        help="To determine radial velocity. default=%(default)g")
    parser.add_argument("--dont-plot", dest='plot', action='store_false', default=True,
                        help='do not display spectrum')
    parser.add_argument('--store-csv', metavar='table.txt',
                        help='store fit results as csv file.')
    return parser


def _sort_by_time(data, times):
    length = len(data)
    assert length == len(times)

    data = [(times[i], data[i]) for i in range(length)]
    data = sorted(data)
    data = [tpl[1] for tpl in data]
    return data


def _three_phases(phases):
    result = []
    result = result + [phase-1.0 for phase in phases]
    result = result + [phase-0.0 for phase in phases]
    result = result + [phase+1.0 for phase in phases]
    return result


def _three_data(data):
    result = data + data + data
    return result


if __name__ == '__main__':
    main()
