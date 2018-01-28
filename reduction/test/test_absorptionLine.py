from unittest import TestCase

from astropy.modeling.fitting import SimplexLSQFitter
from astropy.time import Time
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import EarthLocation

from reduction.spectrum import load as load_spectrum, load_obs_time

from reduction.stars.algol import Algol, algol_coordinate
from reduction.observers import bernd

from reduction.constants import H_ALPHA

from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.nan_statistics import nan_leastsquare

import glob
import numpy as np
import matplotlib.pyplot as plt

import os.path

import logging
logging.basicConfig(level=logging.DEBUG)


def do_no_test_fit_lines():

    def _find_data():

        here = os.path.abspath(os.path.dirname(__file__))

        while len(here) > 1:
            data = os.path.join(here, 'Algol2017', 'data')
            if os.path.isdir(data):
                return data
            else:
                here = os.path.dirname(here)

    algol = Algol()

    phases = []
    computed_redshifts = []
    fitted_redshifts = []
    redshifts_bars = []

    earth_center = EarthLocation.from_geocentric(0, 0, 0, u.meter)

    for filename in glob.iglob(_find_data() + "/**/*noh2o*.fit", recursive=True):

        date_obs, exposure = load_obs_time(filename)

        radial_velocity = algol.rv_A(date_obs) -\
                          algol_coordinate.radial_velocity_correction(obstime=date_obs, location=earth_center)

        redshift = (radial_velocity / c.c).to(1) * H_ALPHA.to('AA')

        phase_AB = algol.phase_AB(date_obs)
        phase_AB_C = algol.phase_AB_C(date_obs)

        try:
            xs, ys, _ = load_spectrum(filename)
        except:
            logging.error('failed to load %s' % filename)
            continue

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        fitter = SimplexLSQFitter()
        fitter._stat_method = nan_leastsquare  # HACK

        initial = AlgolHAlphaModel(sigma=0.4, redshift=redshift)

        initial.sigma.fixed = True
        initial.redshift.fixed = True

        if initial.sigma.fixed:
            weights = np.sqrt(ys)
        else:
            weights = 1.0

        final = fitter(initial, xs, ys, weights=weights)

        # Make sure we disable masking for the result plot
        final.use_mask = False

        fig = plt.figure(figsize=(10, 6))
        # fig.subplots_adjust(hspace=0.6)

        plot1 = fig.add_subplot(211)
        plot2 = fig.add_subplot(212)

        #            plt.plot(xs, inter(xs), label=repr(inter))
        plot1.plot(xs, final(xs), label=repr(final))
        plot1.plot(xs, ys, label=filename)

        plot1.axvline((H_ALPHA + redshift).to('AA').value, label="predicted redshift=$%.1f \AA$" % redshift.value)
        if not initial.redshift.fixed:
            plot1.axvline(H_ALPHA.to('AA').value + final.redshift[0],
                          label=r"fitted redshift=$%.1f \AA$" % final.redshift[0])

        plot2.plot(xs, (ys - final(xs)), label='measurement - fit')

        plot1.set_xlim(final.get_xlimits())
        plot2.set_xlim(final.get_xlimits())

        # plot1.legend()
        plot2.legend()

        plot2.set_title("AB=%.2f, AB/C=%.2f" % (phase_AB.value, phase_AB_C.value))

        # fig.show()
        fig.savefig(filename+".svg")
        plt.close()

        phases.append(algol.AB.true_anomaly(date_obs).value)
        fitted_redshifts.append(final.redshift[0])
        computed_redshifts.append(redshift.value)

    phases = np.array(phases)
    fitted_redshifts = np.array(fitted_redshifts)
    computed_redshifts = np.array(computed_redshifts)

    y = 0.5 * (fitted_redshifts + computed_redshifts)
    yerr = abs(fitted_redshifts - computed_redshifts)

    plt.bar(phases, yerr, 0.01, y-yerr/2, color='b', align='center')
    plt.scatter(phases, computed_redshifts, label="computed")
    plt.scatter(phases, fitted_redshifts, label="fitted")

    plt.ylabel("redshift in $\AA$")
    plt.xlabel("true anomaly")

    plt.legend()
    plt.show()
