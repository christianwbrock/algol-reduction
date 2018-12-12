"""
Methods for loading spectra and observation times from files.
"""

import os.path
import logging

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.io import fits

from reduction.instrument import convolve_with_box

logger = logging.getLogger(__name__)


class Spectrum:
    """
    Define a one dimensional spectrum with equidistant wavelength values
    """

    def __init__(self, x0, dx, nx, ys, unit=u.AA, filename=None, hdu_nbr=None, observer=None, obs_date=None,
                 exposure=None, resolution=None):

        # converts x0 and dx to Angstrom
        if unit:
            if isinstance(unit, str):
                unit = unit.lower()

            if unit != u.AA and unit != 'aa' and unit != 'angstrom':
                x0 = u.Quantity(x0, unit).to(u.AA).value
                dx = u.Quantity(dx, unit).to(u.AA).value

        self.x0 = x0
        self.dx = dx
        self.nx = nx
        self.ys = np.asarray(ys)
        self.observer = observer
        self.obs_date = obs_date
        self.exposure = exposure
        self.resolution = resolution
        self.filename = filename
        self.hdu_nbr = hdu_nbr

        self.xs = np.asarray([self.x0 + i * self.dx for i in range(self.nx)])

        self.xmin = self.x0
        self.xmax = self.x0 + self.dx * (self.nx - 1)

        self.short_name = ""
        if self.filename:
            self.short_name += os.path.basename(self.filename)
        if self.hdu_nbr:
            self.short_name += "(" + self.hdu_nbr + ")"

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys, left=np.NaN, right=np.NaN)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    def __key(self):
        return tuple(self.xs), tuple(self.ys)

    @classmethod
    def load(cls, filename, indices=slice(1)):
        """
        First tries load_from_fits and when that fails load_from_dat

        Parameters
        ----------
        filename: str
            file name of the spectrum

        indices: slice
            the HDU numbers of the spectra to load

        Returns
        -------
        spectra: Spectrum or list
            Depending on `indices` as singe spectrum or multiple spectra
        """
        try:
            return cls.load_from_fit(filename, indices)
        except Exception as e:
            logger.debug("%s", e)
            return cls.load_from_dat(filename)

    @classmethod
    def load_from_dat(cls, filename, wavelength_column=0, intensity_column=1, remove_zeros=True, guess_units=True):
        """load spectrum from dat file using numpy.loadtxt()

        Parameters
        ----------
        filename : str
            file name of a one dimensional dat file i.e. a spectrum

        intensity_column : int
            column of the intensities

        wavelength_column : int
            column of the wavelength

        remove_zeros : bool
            Any leading or trailing zero intensities are removed.

        guess_units : bool
            If values are in 1000-9999 we assume A, below 1000 nanometer.
        """
        try:
            data = np.loadtxt(filename)
        except ValueError:
            data = np.loadtxt(filename, skiprows=1)

        if not data.ndim == 2:
            raise ValueError('file "%s" contains no 2d-table.' % filename)

        requested_columns = 1 + max(wavelength_column, intensity_column)
        if len(data[0]) < requested_columns:
            raise ValueError('file "{0}" contains less than {1} columns'.format(filename, requested_columns))

        # determine range of non-zero data
        begin = 0
        end = len(data)

        if remove_zeros:

            # remove leading zeros
            while begin < end and data[begin][intensity_column] <= 0:
                begin += 1

            # remove trailing zeros
            while begin < end and data[end - 1][intensity_column] <= 0:
                end -= 1

            if not begin < end:
                raise ValueError('file "{0}" contains only zeros in column {1}'.format(filename, intensity_column))

        xs = data[begin:end, wavelength_column]
        ys = data[begin:end, intensity_column]

        unit = None
        if guess_units:
            if np.nanmax(xs) < 1000:
                unit = u.nm
            elif 1000 <= np.nanmin(xs) and np.nanmax(xs) <= 9999:
                unit = u.AA

        return cls.from_arrays(xs, ys, filename, unit)

    @staticmethod
    def from_arrays(xs, ys, filename=None, unit=None):
        """
        Create spectrum from two arrays

        Parameters
        ----------
        xs : array_like
            the wavelength values of the spectrum have to be increasing and equidistant

        ys : array_like
            the intensity values, one for each wavelength value

        filename : str
            passed to Spectrum

        unit :
            usually nm, AA or None; passed to Spectrum
        """

        if not len(xs) == len(ys):
            raise ValueError("xs and ys differ in length")

        x0 = xs[0]
        nx = len(xs)
        dx = (xs[-1] - xs[0]) / (nx - 1)

        # test equidistance
        if not 0.001 * dx > np.max(np.diff(xs)) - np.min(np.diff(xs)):
            raise ValueError("xs are not equidistant")

        return Spectrum(x0, dx, nx, ys, filename=filename, unit=unit)

    @classmethod
    def load_from_fit(cls, filename, indices=slice(1)):
        """
        Loads one or more spectra from a fits file

        Parameters
        ----------
        filename: str
            file name of the spectrum

        indices: slice
            the HDU numbers of the spectra to load

        Returns
        -------
        spectra: Spectrum or list
            Depending on `indices` as singe spectrum or multiple spectra
        """

        with fits.open(filename) as hdu_list:
            logger.debug("load_from_fit(%s)", filename)

            result = []

            matching_indices = indices.indices(len(hdu_list))
            for index in range(*matching_indices):

                hdu = hdu_list[index]

                logger.debug("%s", hdu._summary())

                header = hdu.header
                data = hdu.data

                lambda_0 = header["CRVAL1"]
                nbr_meas = header["NAXIS1"]
                delta_lambda = header["CDELT1"]

                # fix case when CRVAL1 does nor reference the first data point
                crpix1 = header.get('CRPIX1')
                if crpix1 and crpix1 != 1:
                    lambda_0 += (crpix1 - 1) * delta_lambda

                unit = header.get("CUNIT1")

                assert len(data) == nbr_meas,\
                    "header field NAXIS1 %d does not match data size %d" % (nbr_meas, len(data))

                obs_date = cls._load_obs_time(hdu)
                exposure = header.get('EXPTIME')
                if exposure:
                    exposure = int(exposure) * u.second

                observer = header.get('OBSERVER')

                resolution = cls._load_resolution(hdu)

                result.append(
                    Spectrum(lambda_0, delta_lambda, nbr_meas, data, unit=unit, filename=filename, hdu_nbr=index,
                             observer=observer, obs_date=obs_date, exposure=exposure, resolution=resolution))

            single_requested = len(range(*(indices.indices(1000)))) == 1
            if single_requested:
                assert len(result) == 1
                result = result[0]

            return result

    @staticmethod
    def _fix_date_str(date_str):
        date_str1 = date_str.strip()
        if date_str1 != date_str:
            logging.warning('ignore unexpected spaces in date string "%s"', date_str)

        date_str2 = date_str1.replace('_', ':')
        if date_str2 != date_str1:
            logging.warning('replace all "_" with ":" in date string "%s"', date_str)

        return date_str2

    @classmethod
    def _load_obs_time(cls, arg):
        """
        Load observation time attribute as float from fits or None if none exists.

        `DATE-OBS`, `TIME-OBS`, `CIV-DATE`, `JD` and `MJD` are tested.

        Parameters
        ----------
        arg : HDU
            single HDU, HDUList or name of a one dimensional fits file i.e. a spectrum

        Returns
        -------
        resol : float or None
            attribute as float or None
        """

        hdr = arg.header

        date_obs = hdr.get('DATE-OBS')
        if date_obs:
            date_obs = cls._fix_date_str(date_obs)

        if date_obs and len(date_obs) >= len('2017-01-01T19:55'):
            obs_date = Time(date_obs, format='isot')

        elif date_obs and len(date_obs) == len('2017-01-01') \
                and hdr.get('TIME-OBS') and len(hdr.get('TIME-OBS') == len('17:55:22')):
            logger.warning('field "DATE-OBS" contains only the date, no time')
            obs_date = Time(date_obs + 'T' + hdr.get('TIME-OBS'), format='isot')

        elif hdr.get('CIV-DATE'):
            obs_date = Time(hdr.get('CIV-DATE'), format='isot')

        elif hdr.get('JD'):
            obs_date = Time(hdr.get('JD'), format='jd')

        elif hdr.get('MJD'):
            obs_date = Time(hdr.get('MJD'), format='mjd')

        else:
            logger.error('missing field obs-date, civ-date, etc')
            obs_date = None

        return obs_date

    @staticmethod
    def _load_resolution(arg):
        """
        Load 'resol' attribute as float from fits or None if none exists.

        Parameters
        ----------
        arg : HDU
            single HDU, HDUList or name of a one dimensional fits file i.e. a spectrum

        Returns
        -------
        resol : float or None
            attribute as float or None
        """

        res = arg.header.get('BSS_ITRP') or arg.header.get("RESOL") or arg.header.get('BSS_ESRP')

        if isinstance(res, str):
            try:
                res = float(res)
            except Exception as e:
                logger.warning("converting resolution: %s", e)
                res = None

        assert isinstance(res, (type(None), float, int))

        return res


def find_minimum(spectrum, min_x, max_x, box_size_AA):
    """
    Find minimum if a spectrum range after applying a box filter

    Parameters
    ----------
    spectrum : Spectrum

    min_x: float
        lower bound where the minimum is looked for

    max_x: float
        upper bound where the minimum is looked for

    box_size_AA: float
        size of the box filter to be used

    Returns
    -------
    minimum : float
        The x value
    """

    boxed = convolve_with_box(spectrum, box_size_AA)
    mask = [min_x <= x <= max_x for x in boxed.xs]
    index = np.nanargmin(boxed.ys[mask])
    return boxed.xs[mask][index]