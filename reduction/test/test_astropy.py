import numpy.testing


from astropy import coordinates as coord

from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve


def test_constants():
    _ = coord.SkyCoord('3h08.2m', '40.96d', frame='fk5')


def test_convolve():
    data = [1, 1, 2, 1, 1]
    kernel = Gaussian1DKernel(1, x_size=3)

    none = convolve(data, kernel, boundary=None)
    _ = convolve(data, kernel, boundary='fill')
    _ = convolve(data, kernel, boundary='wrap')
    extend = convolve(data, kernel, boundary='extend')

    numpy.testing.assert_equal(none.size, extend.size)
