import os

import sys

from reduction.scripts.display_fits_2d import main


def test_regulus():
    filename = '../../../rawpy/data/alpleo.FIT'
    filename = os.path.join(os.path.dirname(__file__), filename)

    sys.argv = ['dummy', filename, '--dont-show-colorbar']
    main()
