# play around w/ voigth models

import logging
from argparse import ArgumentParser
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting

from reduction.utils.ranges import closed_range, union_of_ranges

from reduction.commandline import filename_parser, verbose_parser, get_loglevel, poly_iglob
from reduction.spectrum import Spectrum

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            fromfile_prefix_chars='@',
                            description='Fit a absorption line with a sum of Voigt-profiles. '
                                        'The spectrum needs to be normalized to one.')
    parser.add_argument('--center-wavelength', '-w', dest='wavelength', default=6562.8,
                        help='Initial center wavelength for fit (default: %(default)s)')
    parser.add_argument('--wavelength-range', '-c', dest='ranges', nargs=2, type=float, metavar=('xmin', 'xmax'),
                        action='append', required=False,
                        help='one or more wavelength ranges used for the polynomial fit')
    parser.add_argument('--non-wavelength-range', '-C', dest='non_ranges', nargs=2, type=float,
                        metavar=('xmin', 'xmax'),
                        action='append', required=False,
                        help='one or more wavelength ranges excluded for the polynomial fit')
    degrees = parser.add_mutually_exclusive_group()
    degrees.add_argument('--num-profiles', '-n', dest='num_profiles', default=3, type=int,
                         help='Number of voigt profiles to fit (default: %(default)s)')
    degrees.add_argument('--num-profiles-range', dest='num_profiles', nargs=2, type=int, metavar=('min', 'max'),
                         help='AIC is used to choose the polynomial degree.')
    parser.add_argument("--dont-plot", dest='plot', action='store_false', default=True,
                        help='do not display spectrum')
    parser.add_argument('--store-csv', metavar='table.txt',
                        help='store fit results as csv file.')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    csv_file = open(args.store_csv, 'w') if args.store_csv else None

    for filename in poly_iglob(args.filenames):

        spectrum = Spectrum.load(filename)

        continuum_ranges = closed_range(np.nanmin(spectrum.xs), np.nanmax(spectrum.xs))
        if args.ranges:
            continuum_ranges &= union_of_ranges(args.ranges)

        if args.non_ranges:
            continuum_ranges &= ~ union_of_ranges(args.non_ranges)

        mask = np.array([x in continuum_ranges for x in spectrum.xs])
        xs = spectrum.xs[mask]
        ys = spectrum.ys[mask]

        fitted_model, err = _fit_model(xs, ys, args.num_profiles, args.wavelength)

        for n, v in zip(fitted_model.param_names, fitted_model.parameters):
            logger.info(f'{n} = {v}')

        const_index = 0
        voigt_indices = list(range(1, fitted_model.n_submodels()))

        absorption = np.sum([fitted_model[i].amplitude_L * fitted_model[i].fwhm_L for i in voigt_indices])

        file_basename = basename(filename)
        if csv_file:
            csv_file.write(f'{file_basename}\t{absorption}\t{err}')

        if args.plot:
            fig, ax = plt.subplots()
            ax.set_title(f'{file_basename}; absorption=${absorption:.2f} \pm {err:.2f} \\AA$')
            ax.plot(spectrum.xs, spectrum.ys, label='spectrum')
            ax.plot(spectrum.xs, fitted_model(spectrum.xs), label='fit')

            for i in voigt_indices:
                ax.plot(spectrum.xs, fitted_model[const_index](spectrum.xs) - fitted_model[i](spectrum.xs),
                        label=f'{fitted_model[i].__class__.__name__} #{i}')

            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            plt.show()
            fig.clear()

        if csv_file:
            csv_file.close()


def _create_model(num_profiles, wavelength):
    fwhm = 20
    model = models.Const1D(1)
    for i in range(1, num_profiles+1):
        model = model - models.Voigt1D(x_0=wavelength, fwhm_L=fwhm)
        fwhm /= 2

    # model[0].amplitude.fixed = True

    for i in range(1, num_profiles+1):
        model[i].amplitude_L.bounds = (0.00001, None)
        model[i].fwhm_L.bounds = (0.00001, None)
        model[i].fwhm_G.bounds = (0.00001, None)

    for i in range(2, num_profiles+1):
        model[i].x_0.tied = lambda m: m[1].x_0

    return model


def _calc_aic(num_models, num_data, err):
    """Compute the Akaike Information Criterion
    """
    k = 2 * num_models + 1  # number of model parameters including noise stddev
    n = num_data

    aic = 2 * k + n * np.log(err / n)
    return aic


def _fit_model(xs, ys, num_profiles, wavelength):

    if isinstance(num_profiles, int):
        fitter = fitting.SLSQPLSQFitter()
        model = fitter(_create_model(num_profiles, wavelength), xs, ys, maxiter=1001)
        return model, fitter.fit_info['final_func_val']

    # else
    assert len(num_profiles) == 2

    aic_list = []
    model_list = []
    err_list = []

    for num in range(num_profiles[0], num_profiles[1]+1):
        model, err = _fit_model(xs, ys, num, wavelength)
        model_list.append(model)
        err_list.append(err)
        aic_list.append(_calc_aic(num, len(xs), err))

    idx_of_min_aic = np.argmin(aic_list)
    return model_list[idx_of_min_aic], err_list[idx_of_min_aic]


if __name__ == '__main__':
    main()
