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
        self.xmin = xs[0]
        self.xmax = xs[-1]
        self.count = len(xs)
        self.dx = (self.xmax - self.xmin) / (self.count - 1)

        for i in range(self.count):
            expected = self.xmin + i * self.dx
            actual = xs[i]
            if abs(expected - actual) * 1000 > self.dx:
                raise ValueError("xs are not equidistant")

    def __call__(self, x):
        if isinstance(x, (np.ndarray, list)):
            return np.array([self(xi) for xi in x])

        assert isinstance(x, (float, int)), "unexpected type {}".format(type(x))

        if not (self.xmin <= x <= self.xmax):
            return float('nan')

        ii = (x - self.xmin) / self.dx
        i_ = int(math.floor(ii))
        f = ii - i_

        assert (0.0 <= f <= 1.0)
        assert (0 <= i_ <= self.count - 1)

        if f == 0.0:
            return self.ys[i_]

        assert (0 <= i_ <= self.count - 2)

        x0 = self.ys[i_]
        x1 = self.ys[i_ + 1]

        return f * x1 + (1 - f) * x0
