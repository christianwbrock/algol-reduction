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


def poly_iglob(filenames_or_patterns: Iterable, *, recursive=True, ignore_case=True):
    """
    Read sequence of file names with or without wildcard characters.
    Any wildcards are replaced by matching file names using glob.iglob().

    :param filenames_or_patterns: iterable of strings with or without wildcard characters.
    :param recursive: is passed to glob.iglob
    :param ignore_case: allows case insensitive matching
    :return: an iterator
    """

    import glob
    from itertools import chain

    result = []

    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c

    for item in filenames_or_patterns:
        try:
            if ignore_case:
                item = ''.join(map(either, item))
            result = chain(result, glob.iglob(item, recursive=recursive))
        except TypeError:
            result = chain(result, glob.iglob(item))

    return result


def poly_glob(filenames_or_patterns, *, recursive=True, ignore_case=True):
    return list(poly_iglob(filenames_or_patterns, recursive=recursive, ignore_case=ignore_case))


# --------- arguments passed to astropy Time -----
def time_parser(prefix, description=None, default=None):
    """
    Some times one wants to ask the user for a time.

    As the astropy.Time can usually not be constructed from a single string and as it has no default argument parser ...

    :param prefix: prefix for the arguments
    :param description: Is shown as argument group description
    :param default: for the first parameter
    :return: an argument parser
    """
    from astropy.time import Time

    description = description or "Arguments to create the %s from astropy.time.Time()" % prefix

    res = ArgumentParser(add_help=False)
    value_group = res.add_argument_group(description=description)
    value_group.add_argument('--' + prefix, default=default, metavar='val1',
                             help='(default: %(default)s)' if default else None)
    value_group.add_argument('--' + prefix + '-val2', metavar='val2')
    value_group.add_argument('--' + prefix + '-format', metavar='format',
                             help='one of %s or guessed from val1' % ", ".join(Time.FORMATS.keys()))
    value_group.add_argument('--' + prefix + '-scale', default='utc', metavar='scale',
                             help='one of %s. (default: %%(default)s)' % ", ".join(Time.SCALES))

    return res


def time_delta_parser(prefix, description=None, default=None):
    """
    Some times one wants to ask the user for a time interval.

    astropy.TimeDelta can usually not be constructed from a single string but it has no default argument parser ...

    :param prefix: prefix for the arguments
    :param description: Is shown as argument group description
    :param default: for the first parameter
    :return: an argument parser
    """
    from astropy.time import TimeDelta

    description = description or "Arguments to create the %s from astropy.time.TimeDelta()" % prefix

    res = ArgumentParser(add_help=False)
    value_group = res.add_argument_group(description=description)
    value_group.add_argument('--' + prefix, default=default, type=float, metavar='val1',
                             help='(default: %(default)s)' if default else None)
    value_group.add_argument('--' + prefix + '-val2', metavar='val2')
    value_group.add_argument('--' + prefix + '-format', metavar='format',
                             help='%s (default: %%(default)s)' % " or ".join(TimeDelta.FORMATS.keys()))
    value_group.add_argument('--' + prefix + '-scale',
                             help='one of %s or None.' % ", ".join(TimeDelta.SCALES))

    return res


def get_time_from_args(args, prefix, required=True):
    """
    Create an astropy.time.Time value from argument crated via time_parser.

    :param args: The argument parser result
    :param prefix: the argument used.
    :param required:  Was this argument required?
    :return: An astropy.time.Time value
    """
    from astropy.time import Time

    prefix = prefix.replace('-', '_', -1)

    val_name = prefix
    val2_name = prefix + '_val2'
    format_name = prefix + '_format'
    scale_name = prefix + '_scale'

    val = getattr(args, prefix)
    val2 = getattr(args, val2_name)
    format_ = getattr(args, format_name)
    scale = getattr(args, scale_name)

    # some formats , e.g. jd have problems with strings
    try:
        val = float(val)
    except:
        pass

    if required and not val:
        raise SystemExit("missing argument %s" % val_name)

    if (val2 or format_) and not val:
        raise SystemExit("%s or %s cannot be used without %s" % (val2_name, format_name, val_name))

    if val:
        return Time(val, val2, format=format_, scale=scale)

    return None


def get_time_delta_from_args(args, prefix, required=True):
    """
    Create an astropy.time.TimeDelta value from argument crated via time_delta_parser.

    :param args: The argument parser result
    :param prefix: the argument used.
    :param required:  Was this argument required?
    :return: An astropy.time.TimeDelta value
    """
    from astropy.time import TimeDelta

    prefix = prefix.replace('-', '_', -1)

    val_name = prefix
    val2_name = prefix + '_val2'
    format_name = prefix + '_format'
    scale_name = prefix + '_scale'

    val = getattr(args, prefix)
    val2 = getattr(args, val2_name)
    format_ = getattr(args, format_name)
    scale = getattr(args, scale_name)

    if required and not val:
        raise SystemExit("missing argument %s" % val_name)

    if (val2 or format_) and not val:
        raise SystemExit("%s or %s cannot be used without %s" % (val2_name, format_name, val_name))

    if val:
        return TimeDelta(val, val2, format=format_, scale=scale)

    return None
