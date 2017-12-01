import unittest

from astropy import coordinates as coord

from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve


class MyTestCase(unittest.TestCase):

    def test_constants(self):
        algol = coord.SkyCoord('3h08.2m', '40.96d', frame='fk5')

    def test_convolve(self):
        data = [1, 1, 2, 1, 1]
        kernel = Gaussian1DKernel(1, x_size=3)

        none = convolve(data, kernel, boundary='None')
        fill = convolve(data, kernel, boundary='fill')
        wrap = convolve(data, kernel, boundary='wrap')
        extend = convolve(data, kernel, boundary='extend')

        self.assertEquals(none.size, extend.size)
