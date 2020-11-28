"""A setuptools based setup module.
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'index.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='algol_reduction',
    version='1.0.0b11',
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
        'scipy==1.4.1',  # there is a bug in 1.5.0 .. 1.5.4 in fitting
        'astropy>=3.2.2',
        'astroplan>=0.6',
        'icalendar'
    ],
    setup_requires=[
        'wheel',
        'twine'
    ],
    package_data={
        '': '*.rst',
        'data': '**/*.spec',
# recursive-include data *.spec.txt
# recursive-include data *.cmdline
    },
    entry_points={
            'console_scripts': [
                'fits_display1d=reduction.scripts.display_fits_1d:main',
                'fits_setval=reduction.scripts.fitssetval:main',
                'fits_timeline=reduction.scripts.fits_timeline:main',
                'helio=reduction.scripts.helio:main',
                'normalize_spectrum=reduction.scripts.normalize_spectrum:main',
                'plan_observations=reduction.scripts.plan_observations:main',
                'generate_report=reduction.scripts.generate_report:main',
                'fit_three_voigts=reduction.utils.fit_h_alpha:main',
            ]
    }
)

