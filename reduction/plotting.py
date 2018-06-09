"""
Setup common plot parameters to be used in books or presentations
"""


def setup_presentation_plots(fontsize=12):
    setup_plots(fontsize=fontsize, usetex=False, figsize=(10, 6))


def setup_text_plots():
    setup_plots(fontsize=8, usetex=True)


def setup_plots(fontsize=8, usetex=True, figsize=None):
    """
    This function adjusts matplotlib settings so that all figures in the
    textbook have a uniform format and look.
    """
    import matplotlib
    if figsize:
        matplotlib.rc('figure', figsize=figsize, dpi=150)
    matplotlib.rc('legend', fontsize=fontsize * 3 // 4, handlelength=3)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('text', usetex=usetex)
    matplotlib.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')
