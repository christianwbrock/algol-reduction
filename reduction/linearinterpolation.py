import math

import numpy as np


class LinearInterpolation:
    """
        We often encounter columnar data  where one column defines
        an equidistant range of function arguments with associated
        values in the other columns.
        This class implements a linear interpolation ...
    """

    def __init__(self, xs, ys, dx):
        self.xs = xs
        self.ys = ys
        self.dx = dx

    @classmethod
    def from_spectrum(cls, spectrum):

        return cls(spectrum.xs, spectrum.ys, spectrum.dx)

    @classmethod
    def from_arrays(cls, xs, ys):
        """ raises ValueError when xs are not equidistant.
        """

        if not (len(xs) == len(ys)):
            raise ValueError("argument length mismatch")

        dx = (xs[-1] - xs[0]) / (len(xs) - 1)

        for i, actual in enumerate(xs):
            expected = xs[0] + i * dx

            if abs(expected - actual) * 1000 > dx:
                raise ValueError("xs are not equidistant")

        return cls(xs, ys, dx)

    @property
    def xmin(self):
        return self.xs[0]

    @property
    def xmax(self):
        return self.xs[-1]

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys, left=np.NaN, right=np.NaN)
