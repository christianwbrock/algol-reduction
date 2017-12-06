from unittest import TestCase

import logging
logging.basicConfig(level=logging.DEBUG)

from astropy.modeling.fitting import SimplexLSQFitter
from astropy.time import Time
import astropy.units as u
import astropy.constants as c

from reduction import spectrum

from reduction.stars.algol import Algol
from reduction.observers import bernd

from reduction.constants import H_ALPHA

from reduction.algol_h_alpha_line_model import AlgolHAlphaModel
from reduction.nan_statistics import nan_leastsquare


class TestAbsorptionLine(TestCase):

    def test_fit_lines(self):

        algol = Algol()

        data_list_filename = '../data/data.lst'
        data_list_filename = os.path.abspath(data_list_filename)

        date_dir = os.path.dirname(data_list_filename)

        data_list = np.genfromtxt(data_list_filename, names=True, dtype=None)

        phases = []
        computed_redshifts = []
        fitted_redshifts = []
        redshifts_bars = []

        for data_item in data_list:

            filename = data_item['filename'].decode('utf8')
            datfile = os.path.join(date_dir, filename)
            date_obs = Time(data_item['DATEOBS'].decode('utf8'), format='isot')

            radial_velocity = algol.rv_A(date_obs) - bernd.heliocentric_correction(date_obs, algol.coordinate)
            redshift = (radial_velocity / c.c).to(1) * H_ALPHA.to('AA')

            phase_AB = algol.phase_AB(date_obs)
            phase_AB_C = algol.phase_AB_C(date_obs)

            try:
                xs, ys = spectrum.load_from_dat(datfile)
            except:
                logging.error('failed to load %s' % filename)
                continue

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
