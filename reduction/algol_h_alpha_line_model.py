import os.path

from reduction.fittable_spectrum import FittableSpectrum

import logging

logger = logging.getLogger(__name__)


class AlgolHAlphaModel(FittableSpectrum):
    """
    Fittable model of the H_alpha absorption line of AlgolA
    
    Measurements can be fitted be determining the redshift, scale and measurement error.
    Please note that evaluate generates nan entries outside the definition of the reference spectrum.
    You have to use a Fitter able to handle such nan return values.
    """

    def __init__(self, *args, **kwargs):
        reference_file = os.path.join(os.path.dirname(__file__),
                                      'CONV_R50._L6563_W200._A_p12500g4.0z-0.5t2.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec')

        super(AlgolHAlphaModel, self).__init__(reference_file, *args, **kwargs)
