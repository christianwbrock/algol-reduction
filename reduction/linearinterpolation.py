import math

import numpy as np


class LinearInterpolation:
    """
        We often encounter columnar data  where one column defines
        an equidistant range of function arguments with associated
        values in the other columns.
        This class implements a linear interpolation ...
    """

    def __init__(self, xs, ys):
        """ raises ValueError when xs are not equidistant.
        """

        if not (len(xs) == len(ys)):
            raise ValueError("argument length mismatch")

        # xs are unused by self
        self.xs = xs
        self.ys = ys

        self.dx = (self.xmax - self.xmin) / (len(xs) - 1)

        for i, actual in enumerate(self.xs):
            expected = self.xmin + i * self.dx

            if abs(expected - actual) * 1000 > self.dx:
                raise ValueError("xs are not equidistant")

    @property
    def xmin(self):
        return self.xs[0]

    @property
    def xmax(self):
        return self.xs[-1]

    def __call__(self, x):
        return np.interp(x, self.xs, self.ys, left=np.NaN, right=np.NaN)
