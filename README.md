Welcome to reduction's documentation!
=====================================

**Note: currently scipy dependency is restricted to version 1.4.1.
Versions 1.5.0 to 1.5.4 contain a bug in the fitting algorithms.**

This library was created to apply doppler tomography to the Algol star system.
This is still the main purpose for most of the code.

However, some parts are generic and can be used for any spectrum and star.
This includes:

#### flux normalization using synthetic spectra

    normalize_spectrum --help

Some example synthetic spectra from the Pollux database are included
in the source distribution of the package.

#### planing observations of variable stars

    plan_observations --help

  This scripts generates `ics` file you can import into your calenders.

#### display spectra

    fits_display1d --help
    fits_display2d --help

#### manipulate fits header

    fits_setval --help

#### loading spectra from fits and text files

#### measuring absorption or emission profiles

Links
-----
There are some additional pages about
- [installation](doc/installation.rst) and
- [creating a documentation](doc/create_docs.rst)

Attribution
-----------
There is a publication describing the [flux normalization](http://spektroskopie.vdsastro.de/files/pdfs/Spektrum55.pdf).
If you publish anything using this code, please add a reference to:

    @article{Brock2019,
        author = {Christian Brock},
        title = {{C}ontinuum {N}ormalization using {S}ynthetic {S}pectra}
        journal = {{SPEKTRUM} Journal of the Section Spectroscopy of the Society of German Amateur Astronomers},
        year = {2019},
        month = {Nov},
        volume= {55},
        pages={4-7},
        issn={2363-5894},
        note={\url{{http://spektroskopie.vdsastro.de/files/pdfs/Spektrum55.pdf}}
    }
