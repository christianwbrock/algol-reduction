import pytest

from reduction.spectrum import Spectrum
import astropy.units as u
import numpy as np


def test_unit_conversion():

    spectrum = Spectrum(3000, 10, 101, np.arange(3000, 4000, 10))
    assert spectrum.xmin == pytest.approx(3000)
    assert spectrum.xmax == pytest.approx(4000)

    spectrum = Spectrum(300, 1, 101, np.arange(3000, 4000, 10), unit=u.nm)
    assert spectrum.xmin == pytest.approx(3000)
    assert spectrum.xmax == pytest.approx(4000)

    spectrum = Spectrum(300, 1, 101, np.arange(3000, 4000, 10), unit='nm')
    assert spectrum.xmin == pytest.approx(3000)
    assert spectrum.xmax == pytest.approx(4000)
