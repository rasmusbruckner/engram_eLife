""" Task-Agent Interaction: Interaction between forgetting model and object-based memory task """

import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
import time

# Use interactive matplotlib backend and turn interactive mode on
# This is required for Ubuntu and plot updates
# However, on a Mac with retina display, the resulting plot is blurry.
# Will hopefully be updated in the future.
plt.ion()


def task_agent_int(task, agent, **kwargs):
    """ This function models the interaction between task and agent

    Note that we have two objects: x_0 (novel object) and x_1 (familiar object).
    We model the probability of exploring the familiar object.
    In the model, mu[0] refers to novel object and mu[1] refers to familiar object.

    :param task: Task-object instance
    :param agent: Agent-object instance
    :param kwargs: Optional
                    - df_data: Dataframe containing behavioral data for model fitting
                    - est_plot: Estimation plot with initial values
                    - figure: Figure handle for estimation plot
                    - ax: Figure axis
    :return: llh: Evaluated likelihood
    :return: df_data: Dataframe containing behavioral data for model fitting
    :return: delta: Prediction errors
    :return: rel: Object-relevancy values
    """

    # Optional input
    df_data = kwargs.get('df', None)  # data frame with parameters for evaluation
    est_plot = kwargs.get('est_plot', None)  # estimation plot with initial values
    figure = kwargs.get('figure', None)  # figure handle for estimation plot
    ax = kwargs.get('ax', None)  # figure axis

    # If no data frame is provided, create one with artificial data for simulation
    if df_data is None:

        simulate = True  # in this case, simulate data
        df_data = pd.DataFrame(index=np.arange(task.n_sim * task.n_treatments),
                               columns=['subj_num', 'treatment', 'object', 'exploration', 'mixture'], dtype='float')
        df_data['object'] = np.tile(['acq_obj_1', 'acq_obj_2', 'test_fam', 'test_novel'], task.n_sim)
        df_data['subj_num'] = np.repeat(np.arange(task.n_sim), task.n_treatments)
        df_data['treatment'] = np.repeat(np.arange(task.n_treatments), task.n_sim)
    else:

        # Otherwise don't simulate but estimate
        simulate = False

        # Extract exploration of novel object x = 0
        exp_novel = df_data[df_data['object'] == "test_novel"]['exploration'].values

        # Extract exploration of familiar object x = 1
        exp_fam = df_data[df_data['object'] == "test_fam"]['exploration'].values

        # Compute probability of familiar object exploration
        mu_fam = exp_fam / (exp_fam + exp_novel)

    # Initialize variables
    delta = np.full([task.n_days, task.n_sim], np.nan)  # prediction errors
    e = np.full([task.n_days, task.n_sim], np.nan)  # engram relevancy
    mu = np.full([task.n_sim, 2], np.nan)  # predicted average exploration probability
    llh = np.full(task.n_sim, np.nan)  # log-likelihood
    p_mu = np.full(task.n_sim, np.nan)  # estimated exploration probability
    sim_exp = np.full(task.n_sim, np.nan)  # simulated exploration
    expressibility = np.full([task.n_days, task.n_sim], np.nan)  # expressibility

    # Cycle over simulations
    for i in range(task.n_sim):

        # Extract current condition
        cond = np.unique(df_data[df_data["subj_num"] == i]["treatment"])[0]

        # Cycle over test days
        for t in range(task.n_days):

            # Object acquisition: Fast learning process after positive PE
            if t == 0:

                # Positive prediction error
                agent.compute_delta(1)

            # Simulate reminder cue
            elif t == task.reminder_pres_time and task.reminder_cue:

                # Positive prediction error
                agent.compute_delta(1)

            # Forgetting: Slow learning process about decaying object relevancy
            else:

                # Negative prediction error
                agent.compute_delta(0)

            # Update engram relevancy
            agent.learn()

            # Record prediction error delta and object relevancy
            delta[t, i] = agent.delta
            e[t, i] = agent.e

            # Compute exploration probability
            agent.softmax()
            expressibility[t, i] = agent.mu[0]  # for expressibility, we take the novel object (x_0)

            # Model retrieval tests
            if (cond == 0 and t == task.test_timepoints[int(cond)]) \
                    or (cond == 1 and t == task.test_timepoints[int(cond)]) \
                    or (cond == 2 and t == task.test_timepoints[int(cond)]) \
                    or (cond == 3 and t == task.test_timepoints[int(cond)]):

                # Store predicted, average exploration probability for test trials
                mu[i, :] = agent.mu

                # Compute shape parameters a and b
                agent.shape_params()

                # Simulate exploration, if required
                if simulate:
                    sim_exp[i] = agent.sample_exploration_prob()
                else:

                    # Compute probability of observed exploration probability using the beta distribution
                    p_mu[i] = beta.pdf(mu_fam[i], agent.a, agent.b)

                    # Note: When object relevancy is higher, it yields less exploration of the familiar object
                    # This is like in the explore-exploit trade-off:
                    #   More knowledge -> less exploration (more exploitation)
                    #   and less knowledge -> more exploration (less exploitation)

                    # Translate into log-likelihood for model estimation
                    llh[i] = np.log(p_mu[i])

    if simulate:

        # Record simulated and average exploration probability
        df_data.loc[df_data['object'] == 'test_novel', 'exploration'] = 1-sim_exp  # simulated exploration of x_0
        df_data.loc[df_data['object'] == 'test_novel', 'mixture'] = mu[:, 0]  # consistent w expressibility (for x_0)
        df_data.loc[df_data['object'] == 'test_fam', 'exploration'] = sim_exp  # sim. exploration of x_1
        df_data.loc[df_data['object'] == 'test_fam', 'mixture'] = mu[:, 1]  # expressibility for x_1

    # If requested, plot estimation in real time
    if est_plot:
        est_plot[0].set_ydata(mu_fam)
        est_plot[1].set_ydata(mu[:, 1])
        ax.set_ylim(0, 1)
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.01)

    # Function output
    if simulate:
        return df_data, delta, e, expressibility
    else:
        return llh, delta, e
