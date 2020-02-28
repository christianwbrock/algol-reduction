""" Normalize spectra
"""
import logging
import warnings
from argparse import ArgumentParser
from functools import lru_cache

import numpy as np
from matplotlib import pyplot as plt

from reduction.instrument import convolve_with_gauss
from reduction.spectrum import Spectrum, find_minimum
from reduction.utils.ranges import closed_range, union_of_ranges, LebesgueSet

logger = logging.getLogger(__name__)


class Normalization:
    """
    Return a polynomial of a given degree that best fits the data points
    passed by xs and ys in the x ranges.

    Parameters
    ----------
        xs: array_like, shape(M,)
            x-coordinates of the M sample points ``(xs[i], ys[i])``.

        ys: array_like, shape(M,)
            y-coordinates of the M sample points ``(xs[i], ys[i])``.

        deg: int
            Degree of the fitting polynomial

        continuum_ranges: LebesgueSet
            Defines ranges belonging top the continuum

    Returns
    -------
        polynomial : callable
    """

    def __init__(self, xs, ys, ref, deg, continuum_ranges, ignore_rank_warning=False):
        self.xs = np.asarray(xs)
        self.ys = np.asarray(ys)
        self.ref_ys_is_defined = ref is not None
        self.ref_ys = np.asarray(ref) if ref is not None else np.ones_like(self.ys)
        self.deg = deg
        self.continuum_ranges = continuum_ranges
        self.ignore_rank_warning = ignore_rank_warning

        assert len(self.xs) == len(self.ys)
        assert self.ref_ys is None or len(self.xs) == len(self.ref_ys)
        assert self.deg >= 0
        assert self.continuum_ranges is None or isinstance(self.continuum_ranges, LebesgueSet)

    @property
    @lru_cache(maxsize=None)
    def norm(self):
        # return np.array([self.ys[i] / self.polynomial(self.xs[i]) for i in range(len(self.xs))])
        return self.ys / self.fit

    @property
    @lru_cache(maxsize=None)
    def polynomial(self):
        return np.poly1d(self.params)

    @property
    @lru_cache(maxsize=None)
    def fit(self):
        return self.polynomial(self.xs)

    @property
    @lru_cache(maxsize=None)
    def params(self):
        logger.debug("fit_polynomial to %d values of order %d", len(self.xs), self.deg)

        ys = (self.ys / self.ref_ys)[self.mask]
        xs = self.xs[self.mask]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore' if self.ignore_rank_warning else 'default', category=np.RankWarning)

            params = np.polyfit(xs, ys, self.deg)

        logger.debug("polynomial params: %s", params)

        return params

    @property
    @lru_cache(maxsize=None)
    def mask(self):
        mask = _ranges_to_mask(self.xs, self.continuum_ranges)
        mask &= np.logical_not(np.isnan(self.xs))
        mask &= np.logical_not(np.isnan(self.ys))
        return mask

    @property
    @lru_cache(maxsize=None)
    def aic(self):
        """Compute the Akaike Information Criterion
        """
        k = len(self.params) + 1  # number of model parameters including noise stddev
        n = len(self.xs[self.mask])

        dy = self.ys[self.mask] - self.polynomial(self.xs[self.mask])
        rss = np.sum(np.square(dy))

        aic = 2 * k + n * np.log(rss / n)
        return aic

    @property
    @lru_cache(maxsize=None)
    def snr(self):
        stddev = np.nanstd((self.norm / self.ref_ys)[self.mask])
        snr = 1.0 / stddev
        return snr

    def plot(self, requested_plot=None):

        plot = requested_plot
        if not plot and logger.getEffectiveLevel() < logging.DEBUG:
            fig = plt.figure()
            plot = fig.add_subplot(111)

        if plot:
            xlim = self.__get_xlim(self.xs, self.continuum_ranges, self.ys, self.ref_ys)
            plot.set_xlim(xlim)
            plot.set_ylim(self.__get_ylim(1.3, xlim, self.xs, self.ys, self.ref_ys))

            if self.continuum_ranges and self.continuum_ranges.is_bounded():
                for r in self.continuum_ranges.intervals():
                    plot.axvspan(r[0], r[1], alpha=0.25)

            plot.plot(self.xs, self.ys, label='meas')

            if self.ref_ys_is_defined:
                plot.plot(self.xs, self.ref_ys, label='ref_ys')
                plot.plot(self.xs, self.ys / self.ref_ys, label='meas / ref_ys')

            plot.plot(self.xs, self.fit, label='polynomial${}^{%d}$' % self.deg)
            plot.plot(self.xs, self.norm, label='normalized')

            plot.legend()

            if not requested_plot:
                plt.show()

    @staticmethod
    def __get_xlim(xs, ranges, y1, y2):
        min_x = xs[0]
        max_x = xs[-1]

        if ranges is not None:
            min_x = max(min_x, ranges.lower_bound())
            max_x = min(max_x, ranges.upper_bound())

        for y in [y1, y2]:

            if y is not None:

                for i in range(len(xs)):
                    if not np.isnan(y[i]):
                        min_x = max(xs[i], min_x)
                        break

                for i in range(1, len(xs)):
                    if not np.isnan(y[-i]):
                        max_x = min(xs[-i], max_x)
                        break

        return min_x, max_x

    @staticmethod
    def __get_ylim(scale, xlim, xs, y1, y2):

        mask = _ranges_to_mask(xs, closed_range(*xlim))

        min_y = np.min(y1[mask])
        max_y = np.max(y1[mask])

        if y2 is not None:
            min_y = min(min_y, np.min(y2[mask]))
            max_y = max(max_y, np.max(y2[mask]))

        return min_y / scale, max_y * scale

    def store_as_dat(self, filename):
        """ Store the normalization result using numpy.savetxt
        """
        if self.ref_ys_is_defined:
            header = 'wavelength object reference normalized'
            data = [self.xs, self.ys, self.ref_ys, self.norm]
        else:
            header = 'wavelength object normalized'
            data = [self.xs, self.ys, self.norm]

        data = np.asarray(data)
        data = np.transpose(data)

        np.savetxt(filename, data, header=header)

    def store_as_fits(self, orig_filename, new_filename, index=0):
        """Load fits header from old file, add some cards and store the normalized spectrum."""

        from astropy.io import fits
        with fits.open(orig_filename) as hdu_list:
            hdr = hdu_list[index].header.copy()

        fits.Header()

        hdr.append(('', ''), end=True)
        hdr.append(('HISTORY', 'Normalized w/ algol-reduction.'), end=True)
        hdr.append(('NORM_DEG', len(self.params) - 1), end=True)
        for i in range(self.polynomial.order):
            hdr.append((f'NORM_CO{i}', self.params[i]), end=True)

        hdr.set(keyword='BITPIX', value=-32)  # the value for float32
        hdr.set(keyword='NAXIS', value=1)
        hdr.set(keyword='NAXIS1', value=len(self.xs))
        hdr.set(keyword='CRPIX1', value=1.0)
        hdr.set(keyword='CRVAL1', value=self.xs[0])
        hdr.set(keyword='CDELT1', value=(self.xs[-1] - self.xs[0]) / (len(self.xs) - 1))
        fits.writeto(new_filename, data=(np.array(self.norm, dtype=np.float32)), header=hdr, checksum=True)


def normalization_parser(add_help=False):
    arg_parser = ArgumentParser(add_help=add_help)
    """
    Use this parser as parent parser in client command line scripts.
    """
    arg_parser.add_argument('--ref', '-r', metavar='reference-spectrum',
                            help='An ease source of reference spectra is "The POLLUX Database of Stellar Spectra" '
                                 '<http://pollux.graal.univ-montp2.fr>.'
                                 'Also "iSpec" <http://www.blancocuaresma.com/s/iSpec> '
                                 'can be used as a frontend for stellar models such as as SPECTRUM, Turbospectrum, '
                                 'SME, MOOG, Synthe/WIDTH9 and others.')
    degrees = arg_parser.add_mutually_exclusive_group()
    degrees.add_argument('--degree', '-d', dest='degree', metavar='val', type=int, default=3,
                         help='degree of the polynomial to be fitted (default: %(default)s)')
    degrees.add_argument('--degree-range', dest='degree', nargs=2, type=int, metavar=('min', 'max'),
                         help='AIC is used to choose the polynomial degree.')
    arg_parser.add_argument('--continuum-range', '-c', dest='ranges', nargs=2, type=float, metavar=('xmin', 'xmax'),
                            action='append', required=False,
                            help='one or more wavelength ranges used for the polynomial fit')
    arg_parser.add_argument('--non-continuum-range', '-C', dest='non_ranges', nargs=2, type=float,
                            metavar=('xmin', 'xmax'),
                            action='append', required=False,
                            help='one or more wavelength ranges excluded for the polynomial fit')
    arg_parser.add_argument('--center-minimum', nargs=3, type=float, metavar=('xmin', 'xmax', 'box-size'),
                            help='calculate redshift from minimum between xmin and xmax after applying a box filter')
    arg_parser.add_argument('--convolve-reference', type=float, metavar='stddev',
                            help='convolve reference spectrum with a gauss kernel to fit the spectrum resolution.')
    arg_parser.add_argument('--convolve-spectrum', type=float, metavar='stddev',
                            help='convolve spectrum with a gauss kernel. <HACK>')

    return arg_parser


def normalize_args(spectrum, args, cut=15):
    """
    Normalize spectrum using commandline args from `normalization_parser`.

    Parameters
    ----------
        spectrum: Spectrum
            the spectrum to normalize

        args : dict
            commandline args from `arg_arg_parser`

        cut : int
            the first and last `cut` values of the spectrum are discarded

    Returns
    -------
        result: Normalization
            normalization result
    """
    spectrum = Spectrum.load(spectrum)

    if spectrum and args.convolve_spectrum and args.convolve_spectrum > 0.0:
        spectrum = convolve_with_gauss(spectrum, args.convolve_spectrum)

    ref_spectrum = Spectrum.load(args.ref) if args.ref else None

    spectrum_resolution = spectrum.resolution

    if cut and cut > 0:
        spectrum = Spectrum.from_arrays(spectrum.xs[cut:-cut], spectrum.ys[cut:-cut], spectrum.filename)

    if ref_spectrum:
        if args.convolve_reference:
            if args.convolve_reference > 0.0:
                ref_spectrum = convolve_with_gauss(ref_spectrum, args.convolve_reference)
        elif spectrum_resolution:
            convolve_reference = 0.5 * (ref_spectrum.xmax + ref_spectrum.xmin) / spectrum_resolution / 2.354
            ref_spectrum = convolve_with_gauss(ref_spectrum, convolve_reference)

    xs = spectrum.xs
    ys = spectrum.ys

    ys /= np.nanmax(ys)

    if args.center_minimum and ref_spectrum:
        min_spectrum = find_minimum(Spectrum.from_arrays(xs, ys), *args.center_minimum)
        min_ref = find_minimum(ref_spectrum, *args.center_minimum)

        redshift = min_spectrum - min_ref
        if redshift:
            logger.info(f'Apply redshift of {redshift} tro reference spectrum')
            ref_spectrum = Spectrum.from_arrays(ref_spectrum.xs + redshift, ref_spectrum.ys)

    continuum_ranges = closed_range(np.nanmin(xs), np.nanmax(xs))
    if ref_spectrum:
        continuum_ranges &= closed_range(ref_spectrum.xmin, ref_spectrum.xmax)

    if args.ranges:
        continuum_ranges &= union_of_ranges(args.ranges)

    # BUG: does not work with two entries like -C 1 2 -C 3 4
    if args.non_ranges:
        continuum_ranges &= ~ union_of_ranges(args.non_ranges)

    ref_ys = ref_spectrum(xs) if ref_spectrum else None

    if isinstance(args.degree, int):
        degree = args.degree
    elif isinstance(args.degree, list):
        degree = range(args.degree[0], args.degree[1] + 1)
    else:
        raise ValueError("missing parameter degree or degree range")

    return normalize(xs, ys, ref_ys, degree, continuum_ranges)


def _ranges_to_mask(xs, ranges):
    if ranges:
        assert isinstance(ranges, LebesgueSet)
        return np.array([x in ranges for x in xs])

    else:
        return np.full(len(xs), fill_value=True, dtype=bool)


def normalize(xs, ys, ref_ys, degree_or_range, continuum_ranges):
    if isinstance(degree_or_range, int):
        return Normalization(xs, ys, ref_ys, degree_or_range, continuum_ranges)

    # else
    from collections.abc import Iterable
    assert isinstance(degree_or_range, Iterable)

    normalizations = []

    for deg in degree_or_range:
        normalizations.append(Normalization(xs, ys, ref_ys, deg, continuum_ranges, ignore_rank_warning=True))

    # would like to call np.min(normalizations, key=Normalization.aic)

    aics = [n.aic for n in normalizations]
    idx_of_min_aic = np.argmin(aics)

    return normalizations[idx_of_min_aic]
