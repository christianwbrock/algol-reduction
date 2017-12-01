import os.path

import numpy as np

from astropy.modeling import Fittable1DModel
from astropy.modeling.parameters import Parameter

from reduction.linearinterpolation import LinearInterpolation
from reduction.instrument import convolve_with_gauss
from reduction.constants import H_ALPHA

from reduction import spectrum


class AlgolHAlphaModel(Fittable1DModel):
    """
    Fittable model of the H_alpha absorption line of AlgolA
    
    Measurements can be fitted be determining the redshift, scale and measurement error.
    Please note that evaluate generates nan entries outside the definition of the reference spectrum.
    You have to use a Fitter able to handle such nan return values.
    """

    redshift = Parameter(default=0.0)
    scale = Parameter(default=1.0, min=0.0)
    offset = Parameter(default=0.0, fixed=True)
    sigma = Parameter(default=0.4, min=0.1)
    use_mask = Parameter(default=False, fixed=True)

    _reffile = '../email_Bernd_2016-10-20/Modell-H-alpla-Algol.dat'
    _reffile = os.path.abspath(_reffile)

    reffilename = os.path.basename(_reffile)


    _reference_spectrum = LinearInterpolation(*spectrum.load_from_dat(_reffile))
    _reference_spectrum_cache = {}

    @staticmethod
    def _get_ref(sigma):
        self = AlgolHAlphaModel

        sigma = round(sigma, 2)

        if sigma not in self._reference_spectrum_cache:
            res = convolve_with_gauss(self._reference_spectrum, sigma)
            self._reference_spectrum_cache[sigma] = res

        return self._reference_spectrum_cache[sigma]

    @staticmethod
    def evaluate(x, redshift, scale, offset, sigma, use_mask):
        assert isinstance(redshift, np.ndarray) and 1 == len(redshift)
        assert isinstance(scale, np.ndarray) and 1 == len(scale)
        assert isinstance(offset, np.ndarray) and 1 == len(offset)
        assert isinstance(sigma, np.ndarray) and 1 == len(sigma)

        ref = AlgolHAlphaModel._get_ref(sigma[0])

        x_shifted = x - redshift[0]
        result = offset[0] + scale[0] * ref(x_shifted)

        # only use core and far wings does not really work
        if use_mask[0]:
            off = np.abs(x_shifted - H_ALPHA.value)
            # mask = np.logical_and(2.0 < off, off < 10.0)
            mask = off < 10
            result[mask] = float('nan')

        return result

    def __repr__(self):
        rep = ""
        if not self.redshift.fixed:
            rep += "redshift=$%.1f \AA$ " % self.redshift[0]
        if not self.scale.fixed:
            rep += "scale=$%.2f$ " % self.scale[0]
        if not self.offset.fixed:
            rep += "offset=$%.2f$ " % self.offset[0]
        if not self.sigma.fixed:
            rep += "stddev=$%.2f \AA$" % self.sigma[0]
        return rep

    def get_xlimits(self):
        return (AlgolHAlphaModel._reference_spectrum.xmin - self.redshift[0],
                AlgolHAlphaModel._reference_spectrum.xmax - self.redshift[0])
