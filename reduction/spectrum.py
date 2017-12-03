import logging
logger = logging.getLogger(__name__)

import numpy as np
from astropy.time import Time
from astropy.io import fits
import astropy.units as u


def load_from_dat(filename):
    """load spectrum from dat file using numpy.load_txt()

    Any leading or trailing zero magnitudes are removed.
    """
    data = np.loadtxt(filename)

    if not data.ndim == 2:
        raise ValueError('file "{0}" contains no 2d-table.'.format(filename))

    if not len(data[0]) == 2:
        raise ValueError('file "{0}" contains more than two columns'.format(filename))

    # determine range of non-zero data
    begin = 0
    end = len(data)

    # remove leading zeros
    while begin < end and data[begin][1] == 0:
        begin += 1

    # remove trailing zeros
    while begin < end and data[end-1][1] == 0:
        end -= 1

    if not begin < end:
        raise ValueError('file "{0}" contains only zeros in the second column'.format(filename))

    xs = data[begin:end, 0]
    ys = data[begin:end, 1]

    return xs, ys


def _fix_date_str(date_str):
    date_str1 = date_str.strip()
    if date_str1 != date_str:
        logging.warning('date string contained spaces')

    date_str2 = date_str1.replace('_', ':', 999)
    if date_str2 != date_str1:
        logging.warning('date string contained _ in places where : were expected')

    return date_str2


def load_obs_time(arg):

    if isinstance(arg, str):
        with fits.open(arg) as hdu:
            logger.debug("filename: %s", arg)
            return load_obs_time(hdu)

    if isinstance(arg, list):
        if len(arg) == 0:
            logger.critical("fits file contains no hdu")
            raise ValueError("empty fits file")

        if len(arg) > 1:
            logger.warning("fits file  contains more than one hdu")

        return load_obs_time(arg[0])

    hdr = arg.header

    if hdr.get('DATE-OBS') and len(_fix_date_str(hdr.get('DATE-OBS'))) >= len('2017-01-01T19:55'):
        obs_date = Time(_fix_date_str(hdr.get('DATE-OBS')), format='isot')

    elif hdr.get('DATE-OBS') and len(hdr.get('DATE-OBS')) == len('2017-01-01')\
            and hdr.get('TIME-OBS') and len(hdr.get('TIME-OBS') == len('17:55:22')):
        logger.warning('field obs-date contains only the date, no time')
        obs_date = Time(hdr.get('DATE-OBS') + 'T' + hdr.get('TIME-OBS'), format='isot')

    elif hdr.get('CIV-DATE'):
        obs_date = Time(hdr.get('CIV-DATE'), format='isot')

    else:
        logger.warning('missing field obs-date, civ-date, etc')
        raise ValueError('missing field obs-date, civ-date, etc')

    if hdr.get('EXPTIME'):
        exposure = int(hdr.get('EXPTIME')) * u.second
    else:
        logging.error('missing field exptime')
        exposure = None

    return obs_date, exposure


def load_from_fit(arg):
    """
    Load first spectrum from a fits file.
    :param arg: of a one dimensional fits file i.e. a spectrum
    :return: intensity, wavelengths and wavelength unit of the spectrum
    """

    from astropy.io import fits

    if isinstance(arg, str):
        with fits.open(arg) as hdu:
            logger.debug("filename: %s", arg)
            return load_from_fit(hdu)

    if isinstance(arg, list):
        if len(arg) == 0:
            logger.critical("fits file contains no hdu")
            raise ValueError("empty fits file")

        if len(arg) > 1:
            logger.warning("WARN: {} contains more than two hdus", arg)

        return load_from_fit(arg[0])

    logger.debug("%s", arg._summary())

    header = arg.header
    data = arg.data

    nbr_meas = header["NAXIS1"]
    lambda_0 = header["CRVAL1"]
    delta_lambda = header["CDELT1"]

    unit = header.get("CUNIT1") or "Angstrom"

    assert len(data) == nbr_meas, "header field NAXIS1 %d does not match data size %d" % (nbr_meas, len(data))

    wavelength = np.array([lambda_0 + i * delta_lambda for i in range(nbr_meas)])

    return wavelength, data, unit
