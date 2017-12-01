import numpy as np

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


def load_from_fit(filename):
    """
    :param filename: of a one dimensional fits file i.e. a spectrum  
    :return: intensity, wavelengths and wavelength unit of the spectrum
    """

    from astropy.io import fits

    hdu = fits.open(filename)

    if len(hdu) > 1:
        print("WARN: %s dontains more than two hdus" % filename)

    header = hdu[0].header
    data = hdu[0].data

    nbr_meas = header["NAXIS1"]
    lambda_0 = header["CRVAL1"]
    delta_lambda = header["CDELT1"]

    unit = header.get("CUNIT1") or "Angstrom"

    assert len(data) == nbr_meas, "header field NAXIS1 %d does not match data size %d" % (nbr_meas, len(data))

    wavelength = np.array([lambda_0 + i * delta_lambda for i in range(nbr_meas)])

    return (wavelength, data, unit)
