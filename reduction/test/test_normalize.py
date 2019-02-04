import os.path

import matplotlib
import pytest

from reduction.normalize import normalization_parser, normalize_args, normalize

matplotlib.use('agg')  # disable plot windows


@pytest.mark.parametrize('reference', [
    "",
    os.path.join(os.path.dirname(__file__),
                 "../../data/synth/CONV_R20.5_L6563_W200._A_p9600g4.1z-0.5t2.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec")
])
@pytest.mark.parametrize('degree', [["2", "10"], "1", "3"])
@pytest.mark.parametrize('center', ["", ["6560.0", "6567.0", "1.5"]])
def test_normalize(reference, degree, center):
    spectrum = os.path.join(os.path.dirname(__file__), "../../data/Wega_2017_07_21_01-noh2o.fit")

    cmd = ""
    if reference:
        cmd = "--ref " + reference

    if isinstance(degree, int):
        cmd += " -d " + degree
    elif isinstance(degree, list):
        cmd += " --degree-range " + " ".join(degree)

    if center:
        cmd += " --center-minimum " + " ".join(center)

    args = normalization_parser().parse_args(cmd.split())

    normalization = normalize_args(spectrum, args)

    # force calculation
    assert len(normalization.fit) == len(normalization.xs)
    assert normalization.aic


def test_synthetic_data():
    import numpy as np

    xs = np.linspace(0, 1, 1000)
    err = 0.05
    correct_deg = 4
    params = np.random.random(correct_deg + 1)
    ys = np.poly1d(params)(xs) + np.random.normal(0, err, len(xs))

    normalization = normalize(xs, ys, None, range(0, 31), None)

    # I hope the linear solution will never win
    assert 0 < normalization.deg < 30

    # from matplotlib import pyplot as plt
    # plot = plt.figure().add_subplot(111)
    # normalization.plot(plot)
    # plt.show()
