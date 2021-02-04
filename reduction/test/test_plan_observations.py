import sys
import pytest
from datetime import datetime, timezone

from astropy.time import Time

from reduction.scripts.plan_observations import main


def test_online():
    start = Time(datetime.now(tz=timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0))
    end = start + 10

    sys.argv = ['dummy', '--verbose',
                '--address=Dresden',
                '--target-name=del cep',
                '--start=%s' % start.isot, '--end=%s' % end.isot,
                '--epoch=2427628.86', '--epoch-format=jd', '--period=5.366',
                '--output=/dev/null']
    main()


def test_offline():
    start = Time(datetime.now(tz=timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0))
    end = start + 10

    sys.argv = ['dummy', '--verbose',
                '--earth-coord', '13.0', '51.0',
                '--entry-prefix=delCep', '--sky-coord', '337.29277091d', '58.41519829d',
                '--start=%s' % start.isot, '--end=%s' % end.isot,
                '--epoch=2427628.86', '--epoch-format=jd', '--period=5.366',
                '--output=/dev/null']
    main()


def test_help():
    with pytest.raises(SystemExit, match="0"):
        sys.argv = ['dummy', '--help']
        main()
