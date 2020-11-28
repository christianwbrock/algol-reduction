# play around w/ Voigt profiles

import logging
from argparse import ArgumentParser
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting

from reduction.commandline import filename_parser, verbose_parser, get_loglevel, poly_iglob
from reduction.spectrum import Spectrum
from reduction.utils.ranges import closed_range, union_of_ranges

logger = logging.getLogger(__name__)


def main():
    args = _arg_parser().parse_args()

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

        fitted_model, err = _fit_model(xs, ys, args.num_profiles, args.fix_continuum, args.center_wavelength,
                                       args.max_wavelength_window, args.max_fwhm)

        for n, v in zip(fitted_model.param_names, fitted_model.parameters):
            logger.info(f'{n} = {v}')

        # diffs between astropy 3.* and 4.0
        n_submodels = fitted_model.n_submodels() if callable(fitted_model.n_submodels) else fitted_model.n_submodels
        const_index = 0
        voigt_indices = list(range(1, n_submodels))

        absorption = np.pi / 2 * np.sum([fitted_model[i].amplitude_L * fitted_model[i].fwhm_L for i in voigt_indices])

        file_basename = basename(filename)
        if csv_file:
            csv_file.write(f'{file_basename}\t{absorption}\t{err}\n')

        if args.plot:
            fig, ax = plt.subplots()
            ax.set_title(f'{file_basename}; absorption=${absorption:.2f} \\pm {err:.2f} \\AA$')
            ax.plot(spectrum.xs, spectrum.ys, label='spectrum')
            ax.plot(spectrum.xs, fitted_model(spectrum.xs), label='fit')

            if continuum_ranges and continuum_ranges.is_bounded():
                for r in continuum_ranges.intervals():
                    ax.axvspan(r[0], r[1], alpha=0.25)

            ax.plot(spectrum.xs, fitted_model[const_index](spectrum.xs),
                    label=_label_of(fitted_model[const_index]))
            for i in voigt_indices:
                ax.plot(spectrum.xs, fitted_model[const_index](spectrum.xs) - fitted_model[i](spectrum.xs),
                        label=_label_of(fitted_model[i]))

            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            plt.show()
            # fig.clear()

    if csv_file:
        csv_file.close()


def _arg_parser():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            fromfile_prefix_chars='@',
                            description='Fit a absorption line with a sum of Voigt-profiles. '
                                        'The spectrum needs to be normalized to one.')
    parser.add_argument('--center-wavelength', '-w', type=float, default=6562.8,
                        help='Initial center wavelength for fit (default: %(default)s)')
    parser.add_argument('--max-wavelength-window', type=float,
                        help='Limit wavelength window around center-wavelength.')
    parser.add_argument('--max-fwhm', type=float, help='Limit FWHM, both for Gaussian and Lorentz.')
    degrees = parser.add_mutually_exclusive_group()
    degrees.add_argument('--num-profiles', '-n', dest='num_profiles', default=3, type=int,
                         help='Number of voigt profiles to fit (default: %(default)s)')
    degrees.add_argument('--num-profiles-range', dest='num_profiles', nargs=2, type=int, metavar=('min', 'max'),
                         help='AIC is used to choose the number of Voigt profiles.')
    parser.add_argument("--dont-fix-continuum", dest='fix_continuum', action='store_false', default=True,
                        help="Do not fix continuum at 1.0")
    parser.add_argument('--wavelength-range', '-c', dest='ranges', nargs=2, type=float, metavar=('xmin', 'xmax'),
                        action='append', help='one or more wavelength ranges used for the fit')
    parser.add_argument('--non-wavelength-range', '-C', dest='non_ranges', nargs=2, type=float,
                        metavar=('xmin', 'xmax'), action='append', required=False,
                        help='one or more wavelength ranges excluded for the fit')
    parser.add_argument("--dont-plot", dest='plot', action='store_false', default=True,
                        help='do not display spectrum')
    parser.add_argument('--store-csv', metavar='table.txt',
                        help='store fit results as csv file.')
    return parser


def _label_of(model):
    params = f"{', '.join((f'{k}={v:.3g}' for k, v in zip(model.param_names, model.parameters)))}"
    return f"{model.__class__.__name__}({params})"


def _create_model(num_profiles, fix_continuum, center_wavelength, max_wavelength_window, max_fwhm):
    fwhm = 20.0
    model = models.Const1D(1)
    for i in range(1, num_profiles + 1):
        model = model - models.Voigt1D(x_0=center_wavelength, fwhm_L=fwhm)
        fwhm /= 2

    if fix_continuum:
        model[0].amplitude.fixed = True

    for i in range(1, num_profiles + 1):
        if max_wavelength_window:
            model[i].x_0.bounds = (center_wavelength - max_wavelength_window / 2,
                                   center_wavelength + max_wavelength_window / 2)
        model[i].amplitude_L.bounds = (0.00001, None)
        model[i].fwhm_L.bounds = (0.00001, max_fwhm)
        model[i].fwhm_G.bounds = (0.00001, max_fwhm)

    for i in range(2, num_profiles + 1):
        model[i].x_0.tied = lambda m: m[1].x_0

    return model


def _calc_aic(model, len_data, err):
    """Compute the Akaike Information Criterion
    """

    # count number of free parameters -- add in one for noise stddev
    num_free_params = 1
    for name in model.param_names:
        if not model.fixed[name] and not model.tied[name]:
            num_free_params += 1

    aic = 2 * num_free_params + len_data * np.log(err / len_data)
    return aic


def _fit_model(xs, ys, num_profiles, fix_continuum, center_wavelength, max_wavelength_window, max_fwhm):
    if isinstance(num_profiles, int):
        fitter = fitting.SLSQPLSQFitter()
        model = fitter(_create_model(num_profiles, fix_continuum, center_wavelength, max_wavelength_window, max_fwhm),
                       xs, ys, maxiter=1001)
        return model, fitter.fit_info['final_func_val']

    # else
    assert len(num_profiles) == 2

    aic_list = []
    model_list = []
    err_list = []

    for num in range(num_profiles[0], num_profiles[1] + 1):
        model, err = _fit_model(xs, ys, num, fix_continuum, center_wavelength, max_wavelength_window, max_fwhm)
        model_list.append(model)
        err_list.append(err)
        aic_list.append(_calc_aic(model, len(xs), err))

    idx_of_min_aic = np.argmin(aic_list)
    return model_list[idx_of_min_aic], err_list[idx_of_min_aic]


if __name__ == '__main__':
    main()
