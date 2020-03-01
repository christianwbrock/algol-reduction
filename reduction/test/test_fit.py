import os
import sys

import pytest

from reduction.utils.fit_h_alpha import main


def test_fit_wega_3():
    filename = '../../data/Wega_2017_07_21_01-noh2o-norm.fit'
    filename = os.path.join(os.path.dirname(__file__), filename)

    # sys.argv = ['dummy', filename, '-w', '6562.5', '--dont-plot', '--store-csv', filename + '.csv']
    sys.argv = ['dummy', filename, '--dont-fix-continuum', '--num-profiles', '3', '-c', '6530', '6620', '-C', '6586',
                '6590']
    # sys.argv = ['dummy', filename, '--num-profiles', '3']
    main()


def test_fit_delcel():
    filename = '../../data/delCep-2019-12-05_04-noh2o-norm.fit'
    filename = os.path.join(os.path.dirname(__file__), filename)

    config = '../../data/voigt_args_delCep.txt'
    config = os.path.join(os.path.dirname(__file__), config)
    config = '@' + config

    sys.argv = ['dummy', config, filename, '--num-profiles-range', '1', '5', '--max-fwhm', '30']
    main()


def test_fit_wega_aic():
    filename = '../../data/Wega_2017_07_21_01-noh2o-norm.fit'
    filename = os.path.join(os.path.dirname(__file__), filename)

    # sys.argv = ['dummy', filename, '-w', '6562.5', '--dont-plot', '--store-csv', filename + '.csv']
    # sys.argv = ['dummy', filename, '--num-profiles-range', '1', '5', '--dont-fix-continuum']
    sys.argv = ['dummy', filename, '--num-profiles-range', '1', '5', '-c', '6530', '6620', '-C', '6586', '6590']
    main()


def test_fit_help():
    sys.argv = ['dummy', '--help']
    with pytest.raises(SystemExit):
        main()
