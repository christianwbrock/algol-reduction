import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

from reduction.plotting import setup_presentation_plots
from reduction.stars.algol import *


def plot_comparison():

    setup_presentation_plots()

    fig = plt.figure()
    plot = fig.add_subplot(111)

    now = Time.now()
    than = now + 3 * u.day

    plt.axhline(y=0)

    for var in [unknown, kosmos_himmeljahr, interstellarum, filipe_diaz, gcvs,
                zavala_2010, zavala2010_via_aaron2012, baron2012,
                aavso_my, aavso_self]:
        var.plot(plot, now, than)

    locator = AutoDateLocator()
    plot.set_title('Time of minimum -- several sources')
    plot.xaxis.set_major_locator(locator)
    plot.xaxis.set_major_formatter(AutoDateFormatter(locator))

    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_comparison()
