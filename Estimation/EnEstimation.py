""" Estimation module: This module contains functions to estimate the model """

import numpy as np
from itertools import compress
import random
from scipy.optimize import minimize
from TaskAgent.EnAgent import Agent
from TaskAgent.en_task_agent_int import task_agent_int
import matplotlib.pyplot as plt
from TaskAgent.EnTask import Task


class EeEstimation:
    """ This class specifies the instance variables and methods of the parameter estimation """

    def __init__(self, est_vars):
        """ This function defines the instance variables unique to each instance

                   See EnEstVars for documentation

               :param est_vars: Estimation variables object instance
        """

        # Parameter names for data frame
        self.kappa = est_vars.kappa
        self.beta = est_vars.beta
        self.alpha_neg = est_vars.alpha_neg

        # Select fixed staring points (used if not rand_sp)
        self.kappa_x0 = est_vars.kappa_x0
        self.beta_x0 = est_vars.beta_x0
        self.alpha_neg_x0 = est_vars.alpha_neg_x0

        # Select range of random starting point values (if rand_sp)
        self.kappa_x0_range = est_vars.kappa_x0_range
        self.beta_x0_range = est_vars.beta_x0_range
        self.alpha_neg_x0_range = est_vars.alpha_neg_x0_range

        # Select boundaries for estimation
        self.kappa_bnds = est_vars.kappa_bnds
        self.beta_bnds = est_vars.beta_bnds
        self.alpha_neg_bnds = est_vars.alpha_neg_bnds

        # Free parameter indexes
        self.which_vars = est_vars.which_vars

        # Fixed parameter values
        self.fixed_mod_coeffs = est_vars.fixed_mod_coeffs

        # Other attributes
        self.n_subj = est_vars.n_subj
        self.n_sp = est_vars.n_sp
        self.rand_sp = est_vars.rand_sp
        self.on_est_plot = est_vars.on_est_plot
        self.est_plot = est_vars.est_plot
        self.figure = est_vars.figure
        self.ax = est_vars.ax

    def select_coeffs(self, coeffs):
        """ This function selects the free and fixed parameters

        :param coeffs: Free parameters
        :return: sel_coeffs: Selected free and fixed parameters
        """

        # Initialize parameter list and counters
        sel_coeffs = []
        i = 0

        # Put selected coefficients in list that is used for model estimation
        for key, value in self.which_vars.items():
            if value:
                sel_coeffs.append(coeffs[i])
                i += 1
            else:
                sel_coeffs.append(self.fixed_mod_coeffs[key])

        return sel_coeffs

    def model_estimation(self, df_subj, agent_vars, task):
        """ This function estimates the free parameters of the model

        :param df_subj: Data frame with data of current participants
        :param agent_vars: Agent variables object instance
        :param task: Task-object instance
        :return: results_list: List containing estimates, llh, bic and age group
        """

        # Extract exploration data for online plotting
        exp_fam = df_subj[df_subj['object'] == "test_fam"]['exploration'].values

        # Initialize best parameter value
        min_x = np.nan

        if self.on_est_plot:

            # Turn interactive mode on to enable live plotting on the screen while parameters are estimated
            plt.ion()

            # Create figure
            self.figure, self.ax = plt.subplots(figsize=(4, 5))

            # Data Coordinates
            x = np.arange(len(exp_fam))
            y = np.repeat(0.5, len(exp_fam))
            z = np.repeat(0.5, len(exp_fam))

            # Plot estimation
            self.est_plot = plt.plot(x, z, '-b', y, '--go')

        # Control random number generator for reproducible results
        random.seed(123)

        # Extract free parameters
        values = self.which_vars.values()

        # Select starting points and boundaries
        # -------------------------------------

        # Extract boundaries
        bnds = [self.kappa_bnds, self.beta_bnds, self.alpha_neg_bnds]

        # Select boundaries according to selected free parameters
        bnds = np.array(list(compress(bnds, values)))

        # Initialize with unrealistically high likelihood
        min_llh = np.inf

        # Cycle over starting points
        for r in range(0, self.n_sp):

            if self.rand_sp:

                # Draw starting points from uniform distribution
                x0 = [random.uniform(self.kappa_x0_range[0], self.kappa_x0_range[1]),
                      random.uniform(self.beta_x0_range[0], self.beta_x0_range[1]),
                      random.uniform(self.alpha_neg_x0_range[0], self.alpha_neg_x0_range[1])]
            else:

                # Use fixed starting points
                x0 = [self.kappa_x0, self.beta_x0, self.alpha_neg_x0]

            # Select starting points according to free parameters
            x0 = np.array(list(compress(x0, values)))

            # Estimate parameters
            res = minimize(self.llh, x0, args=(df_subj, agent_vars, task), method='L-BFGS-B', bounds=bnds,
                           options={'disp': False})

            # Extract minimized log likelihood
            f_llh_min = res.fun

            # Check if negative log-likelihood is lower than the previous one and select the lowest
            if f_llh_min < min_llh:
                min_llh = f_llh_min
                min_x = res.x

        # Compute Bayesian information criterion
        n_subj = len(list(set(df_subj['subj_num'])))  # number of subjects
        bic = self.compute_bic(min_llh, sum(self.which_vars.values()), n_subj)

        # Save data to list of results
        min_x = min_x.tolist()
        results_list = list()
        results_list = results_list + min_x
        results_list.append(float(min_llh))
        results_list.append(float(bic))

        return results_list

    def llh(self, coeffs, df, agent_vars, task):
        """ This function computes the cumulated negative log-likelihood of the data under the model

        :param coeffs: Free parameters
        :param df: Data frame of current subject
        :param agent_vars: Agent-variables-object instance
        :param task: Task-object instance
        :return: llh_sum: Cumulated negative log-likelihood
        """

        # Get parameters
        sel_coeffs = self.select_coeffs(coeffs)

        # Model variables
        agent_vars.kappa = sel_coeffs[0]
        agent_vars.beta = sel_coeffs[1]
        agent_vars.alpha_neg = sel_coeffs[2]

        # Call agent-object instance
        agent = Agent(agent_vars)

        # Estimate parameters
        if self.on_est_plot:
            llh_mix, _, _ = task_agent_int(task, agent, df=df, figure=self.figure, ax=self.ax, est_plot=self.est_plot)
        else:
            llh_mix, _, _ = task_agent_int(task, agent, df=df)

        # Compute cumulated negative log-likelihood
        llh_sum = -1 * np.sum(llh_mix)

        return llh_sum

    @staticmethod
    def compute_bic(llh, n_params, n_trials):
        """ This function computes the Bayesian information criterion (BIC)

            See Stephan et al. (2009). Bayesian model selection for group studies. NeuroImage

        :param llh: Negative log-likelihood
        :param n_params: Number of free parameters
        :param n_trials: Number of trials
        :return: bic
        """

        return (-1 * llh) - (n_params / 2) * np.log(n_trials)


def estimation_func(est_vars, agent_vars, df_data):
    """ This function starts the estimation of parameters

        This is more efficient that calling each step multiple times.

    :param est_vars: Estimation-variables-object instance
    :param agent_vars: Agent-variables-object instance
    :param df_data: Data frame for estimation
    :return: estimates: Parameter estimates
    """

    # Number of subjects
    n_subj = len(list(set(df_data['subj_num'])))

    # Estimation-object instance
    ee_estimation = EeEstimation(est_vars)

    # Get task-object instance
    task = Task()
    task.n_sim = n_subj

    # Reset index of data frame
    df_data = df_data.reset_index(drop=True)

    # Replace actual ID with subject number within current group (therefore not done in preprocessing)
    df_data['subj_num'] = np.repeat(np.arange(n_subj), 5)

    # Estimate the parameters
    estimates = ee_estimation.model_estimation(df_data, agent_vars, task)

    return estimates
