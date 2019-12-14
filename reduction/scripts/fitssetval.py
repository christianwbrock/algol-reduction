#!python
# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser

from astropy.io import fits

from reduction.commandline import filename_parser, poly_glob


def main():
    parser = ArgumentParser(parents=[filename_parser('fits-file')])
    parser.add_argument("--kvc", nargs=3, type=str, metavar=('header', 'value', 'comment'),
                        action='append', help='set header entry to all fits files')
    parser.add_argument("--kv", nargs=2, dest='kvc', type=str, metavar=('header', 'value'),
                        action='append', help='set header entry to all fits files')
    parser.add_argument("--backup", default=False, action='store_true', help="save a backup file")
    parser.add_argument("-e", "--extension", default=['*'], action='append',
                        help="Either a commas separated list of extension numbers or a '*' for all")

    args = parser.parse_args()

    if not args.kvc or not len(args.kvc):
        sys.exit('missing required argument --kv or --kvc')

    filenames = poly_glob(args.filenames)
    if not filenames:
        sys.exit('the passed filename pattern yield no file')

    for filename in filenames:

        with fits.open(filename, mode='update', save_backup=(args.backup)) as content:
            for i, hdu in enumerate(content, 0):

                if '*' not in args.extension and i not in args.extension and str(i) not in args.extensions:
                    continue

                header = hdu.header

                for kvc in args.kvc:
                    assert 2 <= len(kvc) <= 3

                    key = kvc[0]
                    value = kvc[1]
                    comment = kvc[2] if 3 == len(kvc) else None

                    header.set(keyword=key, value=value, comment=comment)


if __name__ == "__main__":
    main()
