import os
import sys

from reduction.utils.fit_h_alpha import main
import pytest

def test_fit_wega_3():
    filename = '../../data/Wega_2017_07_21_01-noh2o-norm.fit'
    filename = os.path.join(os.path.dirname(__file__), filename)

    # sys.argv = ['dummy', filename, '-w', '6562.5', '--dont-plot', '--store-csv', filename + '.csv']
    sys.argv = ['dummy', filename, '--num-profiles', '3']
    main()


def test_fit_wega_aic():
    filename = '../../data/Wega_2017_07_21_01-noh2o-norm.fit'
    filename = os.path.join(os.path.dirname(__file__), filename)

    # sys.argv = ['dummy', filename, '-w', '6562.5', '--dont-plot', '--store-csv', filename + '.csv']
    sys.argv = ['dummy', filename, '--num-profiles-range', '1', '5']
    main()


def test_fit_help():
    sys.argv = ['dummy', '--help']
    with pytest.raises(SystemExit):
        main()
