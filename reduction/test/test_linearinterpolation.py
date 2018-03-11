#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import random

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import pytest

from reduction.linearinterpolation import LinearInterpolation


@pytest.mark.parametrize('args', [
    [[1, 2, 3], [7, 8]],  # length error
    [[1, 2], [7, 8, 9]],  # length error
    [[1, 2, 4], [7, 8, 9]],  # not equidistant
])
def test_arg_error_len(args):
    try:
        LinearInterpolation.from_arrays(args[0], args[1])
        assert False, 'ValueError expected'
    except ValueError:
        pass


def test_apply1():
    f = LinearInterpolation.from_arrays([0, 1], [0, 1])

    for x in (random() for i in range(200)):
        assert_equal(f(x), x)


def test_apply2():
    xs = np.linspace(-1.0, 1.0, 21)
    ys = np.sin(xs)

    f = LinearInterpolation.from_arrays(xs, ys)

    assert_equal(-1.0, f.xmin)
    assert_equal(+1.0, f.xmax)

    for x in np.linspace(-1, 1, 201):
        expected = np.sin(x)
        actual = f(x)

        assert_almost_equal(expected, actual, 1)

    assert (np.isnan(f(-1.1)))
    assert (np.isnan(f(+1.1)))


def test_list():
    f = LinearInterpolation.from_arrays([0, 1], [0, 1])
    assert_equal([0, 0.5, 1], f([0, 0.5, 1]))
