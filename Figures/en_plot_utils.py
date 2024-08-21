# This document contains plot utilities

def latex_plt(matplotlib):
    """ This function updates the matplotlib library to use Latex and changes some default plot parameters

    :param matplotlib: matplotlib instance
    :return: Updated matplotlib instance
    """

    pgf_with_latex = {
        "axes.labelsize": 6,
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "pgf.rcfonts": False,
    }
    matplotlib.rcParams.update(pgf_with_latex)

    return matplotlib