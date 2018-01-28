"""
Collections of tools helpful to implement __main__ methods.
"""

from typing import Iterable
from argparse import ArgumentParser
from logging import Logger

# -------------------- verbose --------------------
verbose_parser = ArgumentParser(add_help=False)
_verbose_group = verbose_parser.add_mutually_exclusive_group()
_verbose_group.add_argument('-v', '--verbose', action='count', default=0, help='increment verbosity level')
_verbose_group.add_argument('-q', '--quiet', action='count', default=0, help='decrement verbosity level')


def get_loglevel(logger_or_level, args):
    """
    Apply verbose arguments passed to verbose_parser to determine a log level.

    :param logger_or_level:
    :param args:
        A ArgumentParser namespace containing verbose and quiet members.
    :return:
        the newly calculated log level that can be passed to logging.basicConfig()
    """

    if isinstance(logger_or_level, Logger):
        logger_or_level = logger_or_level.getEffectiveLevel()

    level = max(0, logger_or_level - 10 * args.verbose + 10 * args.quiet)
    return level


# -------------- required filenames ---------------
def filename_parser(metavar: str):
    """ arg_parser to gather one or more filenames in an os independent way.

    This is my approach to ensure that # python main.py *.txt *.ini works consistently on windows and posix machines.
    On windows systems the command shells don't do wildcard expansion, the wildcard is passed to the application.

    :param metavar: passed as ArgumentParser.metavar to be displayed in the help message.
    """

    parser = ArgumentParser(add_help=False)
    """
    Use this parser as parent parser in client command line scripts.
    """
    parser.add_argument('filenames', nargs='+', metavar=metavar, help='one or more ' + metavar + ' files')

    return parser


def poly_iglob(filenames_or_patterns: Iterable, *, recursive=True):
    """
    Read sequence of file names with or without wildcard characters.
    Any wildcards are replaced by matching file names using glob.iglob().

    :param filenames_or_patterns: iterable of strings with or without wildcard characters.
    :param recursive: is passed to glob.iglob
    :return: an iterator
    """

    import glob
    from itertools import chain

    result = []

    for item in filenames_or_patterns:
        try:
            result = chain(result, glob.iglob(item, recursive=recursive))
        except TypeError:
            result = chain(result, glob.iglob(item))

    return result


def poly_glob(filenames_or_patterns, *, recursive=True):
    return list(poly_iglob(filenames_or_patterns, recursive=recursive))


# --------- arguments passed to astropy Time -----
def time_parser(prefix):
    from astropy.time import Time

    res = ArgumentParser(add_help=False)
    excl_group = res.add_mutually_exclusive_group()
    value_group = excl_group.add_argument_group()
    value_group.add_argument('--' + prefix, help='passed as first parameter to atropy.time.Time()')
    value_group.add_argument('--' + prefix + '-val2', help='passed as second parameter to atropy.time.Time()')
    value_group.add_argument('--' + prefix + '-format', choices=Time.FORMATS,
                             help='passed as format parameter to atropy.time.Time()')
    excl_group.add_argument('--' + prefix + '-jd', type=int, help='time in JD format')
    excl_group.add_argument('--' + prefix + '-mjd', type=int, help='time in MJD format')

    return res


def get_time_from_args(args, prefix, required=True):
    from astropy.time import Time

    prefix = prefix.replace('-', '_', -1)

    jd_name = prefix + '_jd'
    mjd_name = prefix + '_mjd'
    val_name = prefix
    val2_name = prefix + '_val2'
    format_name = prefix + '_format'

    jd = getattr(args, jd_name)
    mjd = getattr(args, mjd_name)
    val = getattr(args, prefix)
    val2 = getattr(args, val2_name)
    format_ = getattr(args, format_name)

    if required and not (jd or mjd or val):
        raise SystemExit("missing argument %s, %s  or %s" % (val_name, jd_name, mjd_name))

    def raise_if_both(param1, param2, name1, name2):
        if param1 and param2:
            raise SystemExit("argument %s not allowed with argument %s" % (name1, name2))

    raise_if_both(jd, mjd, jd_name, mjd_name)
    raise_if_both(jd, val, jd_name, val_name)
    raise_if_both(mjd, val, mjd_name, val_name)

    if (val2 or format_) and not val:
        raise SystemExit("%s or %s cannot be used without %s" % (val2_name, format_name, val_name))

    if jd:
        return Time(jd, format='jd')

    elif mjd:
        return Time(mjd, format='mjd')

    elif val:
        return Time(val, val2, format=format_)

    return None
