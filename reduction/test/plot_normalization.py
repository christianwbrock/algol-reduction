from unittest import TestCase

import numpy as np

from reduction.spectrum import load_from_fit
from reduction.normalize import normalize, plot_normalized, fit_polynomial

import matplotlib.pyplot as plt
from os.path import basename


class TestNormalization(TestCase):

    def test_normalization(self):

        filenames = ["../data/Periode_2016_2017/Bernd_Bitnar_RC16/algol_2017_03_27_02-noh2o.fit",
                     "../data/Periode_2017_2018/Ulrich_Waldschläger/Algol_170828-1800_noh2o_uw.fit",
                     "../data/Periode_2017_2018/Uwe_Zurmuehl/Algol_AT1B1No_TOGi+Blk_170913_C1_ED80_434_9m60s_AAMs2Mul4Crp_LTs2b1Ha_6236-7029l-noh2o.FITS"]

        degrees = [3, 5]
        ranges = [[6500, 6510], [6520, 6530], [6590, 6650]]

        range_min = ranges[0][0]
        range_max = ranges[-1][-1]
        range_size = range_max - range_min

        for i, filename in enumerate(filenames):
            ax = plt.subplot(len(filenames), 1, i + 1)

            plot_normalized(filename, ranges, degrees, ax)

            ax.set_xlim([range_min - 0.2 * range_size, range_max + 0.2 * range_size])
            ax.set_ylim([-0.1, 1.1])
            ax.set_title(filename)
            ax.legend()

        plt.show()
