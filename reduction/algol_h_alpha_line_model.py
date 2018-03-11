import os.path

import numpy as np

from astropy import units as u
from astropy.units import Quantity
from astropy.modeling import Fittable1DModel
from astropy.modeling.parameters import Parameter

from reduction.linearinterpolation import LinearInterpolation
from reduction.instrument import convolve_with_gauss
from reduction.constants import H_ALPHA

from reduction.spectrum import Spectrum

from functools import lru_cache

import logging
logger = logging.getLogger(__name__)


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

    def __init__(self, *args, **kwargs):

        arg_units = {'redshift': u.AA, 'sigma': u.AA}

        for k, v in kwargs.items():
            if k in arg_units and isinstance(v, Quantity):
                kwargs[k] = v.to(arg_units[k]).value

        super(AlgolHAlphaModel, self).__init__(*args, **kwargs)

    #
    # _reference_file = os.path.join(os.path.dirname(__file__), 'Modell-H-alpla-Algol.dat')
    _reference_file = os.path.join(os.path.dirname(__file__), '../data/synth/algol-kolbas2005.fit')
    _reference_spectrum = LinearInterpolation.from_spectrum(Spectrum.load(_reference_file))

    @classmethod
    @lru_cache(maxsize=None)
    def __get_ref_from_cache(cls, sigma):
        logger.debug("convolve with sigma=%g", sigma)
        return convolve_with_gauss(cls._reference_spectrum, sigma)

    @classmethod
    def _get_ref(cls, sigma):
        return cls.__get_ref_from_cache(round(sigma, 3))

    @classmethod
    def evaluate(cls, x, redshift, scale, offset, sigma):
        assert isinstance(redshift, np.ndarray) and 1 == len(redshift)
        assert isinstance(scale, np.ndarray) and 1 == len(scale)
        assert isinstance(offset, np.ndarray) and 1 == len(offset)
        assert isinstance(sigma, np.ndarray) and 1 == len(sigma)

        sigma = sigma[0]
        redshift = redshift[0]
        scale = scale[0]
        offset = offset[0]

        logger.debug("evaluate model: redshift=%g, sigma=%g, ...", redshift, sigma)

        ref = cls._get_ref(sigma)

        x_shifted = x - redshift

        result = offset + scale * ref(x_shifted)

        return result

    def __str__(self):
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
        return AlgolHAlphaModel.get_xlimit_for_redshift(self.redshift[0])

    @classmethod
    def get_xlimit_for_redshift(cls, redshift):
        return (cls._reference_spectrum.xmin - redshift,
                cls._reference_spectrum.xmax - redshift)

    @classmethod
    def plot(cls, plot):
        plot.plot(cls._reference_spectrum.xs, cls._reference_spectrum.ys)
