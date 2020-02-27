import os
import sys

from reduction.utils.fit_h_alpha import main


def test_fit_wega():
    filename = '../../data/Wega_2017_07_21_01-noh2o-norm.fit'
    filename = os.path.join(os.path.dirname(__file__), filename)

    # sys.argv = ['dummy', filename, '-w', '6562.5', '--dont-plot']
    sys.argv = ['dummy', filename, '--store-csv', filename + '.csv']
    main()
