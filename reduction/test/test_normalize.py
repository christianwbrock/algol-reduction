from random import random

import os.path
import pytest

from reduction.normalize import normalize_spectrum
from reduction.spectrum import Spectrum

from matplotlib import pyplot as plt


@pytest.mark.parametrize('reference', [
    None,
    Spectrum.load(os.path.join(os.path.dirname(__file__),
                               "../../data/synth/CONV_R20.5_L6563_W200._A_p9600g4.1z-0.5t2.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec"))
])
@pytest.mark.parametrize('degree', [1, 3])
@pytest.mark.parametrize('method', ['hermit', 'polynomial', None])
@pytest.mark.parametrize('center', [None, [6560.0, 6567.0, 1.5]])
def test_normalize(reference, degree, method, center):
    spectrum = Spectrum.load(os.path.join(os.path.dirname(__file__), "../../data/Wega_2017_07_21_01-noh2o.fit"))
    plot = plt.axes()
    normalize_spectrum(spectrum, reference, degree=degree, method=method, center_minimum=center, requested_plot=plot)
    # plt.show()
    plt.close()
