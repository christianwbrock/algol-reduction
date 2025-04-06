from reduction.algol_h_alpha_line_model import AlgolHAlphaModel

import numpy as np


def test_model():
    h_alpha = AlgolHAlphaModel()

    xmin, xmax = h_alpha.get_xlimits()

    assert np.isnan(h_alpha(xmin - 1))
    assert np.isnan(h_alpha(xmax + 1))

    xmin += 3 * h_alpha.sigma
    xmax -= 3 * h_alpha.sigma

    assert not np.isnan(h_alpha(xmin))
    assert not np.isnan(h_alpha(xmax))

    res = h_alpha(np.linspace(xmin, xmax))
    assert np.logical_and(0.0 <= res, res <= 1.0).all()
