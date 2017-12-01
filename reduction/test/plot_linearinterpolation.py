#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from reduction.linearinterpolation import LinearInterpolation


def main():
    x10 = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    f = LinearInterpolation(x10, np.sin(x10))

    x100 = np.linspace(0, 2 * np.pi, 100, endpoint=True)
    f100 = f(x100)
    s100 = np.sin(x100)

    plt.plot(x100, s100, label='sin')
    plt.plot(x100, f100, label='lineare interpolation')
    plt.legend(loc='upper right')
    plt.show()

    return 0


if __name__ == '__main__':
    main()
