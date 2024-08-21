""" Estimation variables: This class contains the estimation variables """

import numpy as np


class EeEstVars:
    # This class defines the instance variables unique to each instance

    def __init__(self):
        # This function determines the default estimation variables

        # Parameter names for data frame
        self.kappa = 'kappa'
        self.beta = 'beta'
        self.alpha_neg = 'alpha_neg'

        # Select fixed staring points (used if not rand_sp)
        self.kappa_x0 = 20
        self.beta_x0 = -0.6
        self.alpha_neg_x0 = 0.05

        # Select range of random starting point values (if rand_sp)
        self.kappa_x0_range = (1, 100)
        self.beta_x0_range = (-10, 0)
        self.alpha_neg_x0_range = (0, 1)

        # Select boundaries for estimation
        self.kappa_bnds = (1, 100)
        self.beta_bnds = (-10, 0)
        self.alpha_neg_bnds = (0, 0.25)

        # Free parameter indexes
        self.which_vars = {self.kappa: True,
                           self.beta: True,
                           self.alpha_neg: True
                           }

        # Fixed parameter values
        self.fixed_mod_coeffs = {self.kappa: 20,
                                 self.beta: -0.6,
                                 self.alpha_neg: 0.1
                                 }

        # Other attributes
        self.n_subj = np.nan  # number of subjects
        self.n_sp = 10  # number of starting points
        self.rand_sp = True  # turn random starting points on and off
        self.on_est_plot = True  # online illustration of fitting
        self.est_plot = np.nan  # estimation-plot object
        self.figure = np.nan  # figure object of online plot
        self.ax = np.nan  # axis object of online plot
