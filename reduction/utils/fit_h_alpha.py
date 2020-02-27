# play around w/ voigth models

import logging
from argparse import ArgumentParser
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting

from reduction.commandline import filename_parser, verbose_parser, get_loglevel, poly_iglob
from reduction.spectrum import Spectrum

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(parents=[filename_parser('spectrum'), verbose_parser],
                            fromfile_prefix_chars='@',
                            description='Fit a absorption line with a sum of three Voigt-profiles. '
                                        'The spectrum needs to be normalized to one.')

    parser.add_argument('--wavelength', '-w', default=6562.8, help='Initial wavelength for fit.')

    parser.add_argument("--dont-plot", dest='plot', action='store_false', default=True,
                        help='do not display spectrum')

    parser.add_argument('--store-csv', metavar='table.txt',
                        help='store fit results as csv file.')

    args = parser.parse_args()

    logging.basicConfig(level=get_loglevel(logger, args))

    csv_file = open(args.store_csv, 'w') if args.store_csv else None

    for filename in poly_iglob(args.filenames):

        spectrum = Spectrum.load(filename)

        initial_model = (models.Const1D(1) -
                         models.Voigt1D(x_0=args.wavelength, fwhm_L=20) -
                         models.Voigt1D(x_0=args.wavelength, fwhm_L=2) -
                         models.Voigt1D(x_0=args.wavelength, fwhm_L=1))

        const_index = 0
        voigt_indices = (1, 2, 3)

        # make sure all lines have the same wavelength
        initial_model[3].x_0.tied = lambda model: model[1].x_0
        initial_model[2].x_0.tied = lambda model: model[1].x_0

        fitter = fitting.SLSQPLSQFitter()
        fitted_model = fitter(initial_model, spectrum.xs, spectrum.ys, maxiter=1001)

        err = fitter.fit_info['final_func_val']

        for n, v in zip(fitted_model.param_names, fitted_model.parameters):
            logger.info(f'{n} = {v}')

        absorption = np.sum([fitted_model[i].amplitude_L * fitted_model[i].fwhm_L for i in voigt_indices])

        file_basename = basename(filename)
        if csv_file:
            csv_file.write(f'{file_basename}\t{absorption}')

        if args.plot:
            plt.subplots(1, 1)
            plt.subplot(1, 1, 1)
            plt.title(f'{file_basename}; absorption=${absorption:.4g} \\AA$')
            plt.plot(spectrum.xs, spectrum.ys, label='spectrum')
            plt.plot(spectrum.xs, fitted_model(spectrum.xs), label=f'fit: residual={err:.2g}')

            for i in voigt_indices:
                plt.plot(spectrum.xs, fitted_model[const_index](spectrum.xs) - fitted_model[i](spectrum.xs),
                         label=f'{fitted_model[i].__class__.__name__} #{i}')

            plt.ylim(0, 1)
            # plt.xlim(6563 - 20, 6563 + 20)

            plt.legend()
            plt.show()
            plt.clf()

        if csv_file:
            csv_file.close()


if __name__ == '__main__':
    main()
