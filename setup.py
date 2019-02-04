"""A setuptools based setup module.
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'index.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='reduction',
    version='1.0.0a8',
    description='Spectral reduction package',
    long_description=long_description,
    author='Christian W. Brock',
    license='BSD',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'astropy',
        'astroplan',
        'icalendar'
    ],
    package_data={
        'reduction': ['*.spec', '*.spec.txt']
    },
    scripts=[
        'bin/display_fits_1d.py',
        'bin/fits_timeline.py',
        'bin/normalize_spectrum.py',
        'bin/star_altitude.py',
        'bin/generate_report.py',
        'bin/plan_observations.py',
        'bin/fitssetval.py'
    ]
)
