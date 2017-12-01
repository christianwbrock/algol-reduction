#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from random import random

import numpy as np


from reduction.linearinterpolation import LinearInterpolation


class LinearInterpolationTest(unittest.TestCase):
    def test_arg_error_equidist(self):
        try:
            LinearInterpolation([1, 2, 4], [7, 8, 9])
            self.fail("Value Error (about equidistance) was expected")
        except ValueError:
            pass

    def test_arg_error_len(self):
        try:
            LinearInterpolation([1, 2, 3], [7, 8])
            self.fail("Value Error (about length differences) was expected")
        except ValueError:
            pass

    def test_apply1(self):
        f = LinearInterpolation([0, 1], [0, 1])

        for x in (random() for i in range(200)):
            self.assertEqual(f(x), x)

    def test_apply2(self):
        xs = [(-1.0 + 0.1 * i) for i in range(21)]
        ys = np.sin(xs)

        f = LinearInterpolation(xs, ys)

        self.assertEquals(-1.0, f.xmin)
        self.assertEquals(+1.0, f.xmax)
        self.assertEquals(21, f.count)

        for x in ((-1.0 + 0.01 * i) for i in range(201)):
            expected = np.sin(x)
            actual = f(x)

            assert (np.fabs(expected - actual) < 0.1)

        assert (np.isnan(f(-1.1)))
        assert (np.isnan(f(+1.1)))

    def test_list(self):
        f = LinearInterpolation([0, 1], [0, 1])
        self.assertEquals([0, 0.5, 1], f([0, 0.5, 1]).tolist())
