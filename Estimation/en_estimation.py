""" This script runs the model-estimation pipeline

    1. Load data
    2. Prepare analysis
    3. Estimate model for enrichment experiment
    4. Estimate model for Rac1-experiment
"""


if __name__ == '__main__':

    import numpy as np
    from scipy.stats import beta
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    from EnEstVars import EeEstVars
    from Estimation.EnEstimation import estimation_func
    from TaskAgent.EnAgentVars import AgentVars
    from en_utilities import get_group_data, merge_estimates, safe_save_dataframe, prepare_out_of_sample
    from Figures.en_plot_utils import latex_plt

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # -------------
    # 1. Load data
    # -------------

    df_enrichment_cont, df_enrichment_exp = get_group_data("en_data/en_prepr_data_ee.pkl")
    df_rac1_cont, df_rac1_exp = get_group_data("en_data/en_prepr_data_rac1.pkl")

    # -------------------
    # 2. Prepare analysis
    # -------------------

    agent_vars = AgentVars()
    est_vars = EeEstVars()
    est_vars.rand_sp = True
    est_vars.n_sp = 3  # 10

    # -------------------------------------------
    # 3. Estimate model for enrichment experiment
    # -------------------------------------------

    # Estimate free parameters
    df_estimates_cont = estimation_func(est_vars, agent_vars, df_enrichment_cont)  # control group
    df_estimates_exp = estimation_func(est_vars, agent_vars, df_enrichment_exp)  # experimental group

    # Number of subjects, corresponding to trials in our case (one animal = one trial)
    n_subj_cont = len(list(set(df_enrichment_cont['subj_num'])))
    n_subj_exp = len(list(set(df_enrichment_exp['subj_num'])))

    # Compute BIC for random model (control model): We assume 50% exploration, which might be a bit arbitrary
    # In random model, there are no free parameters, so no "(n_params / 2) * np.log(n_trials)" term
    # and basically just sum of log-likelihoods
    p_rand = beta.pdf(0.5, 1, 1)
    bic_rand_cont = np.sum(np.repeat(np.log(p_rand), n_subj_cont))
    bic_rand_exp = np.sum(np.repeat(np.log(p_rand), n_subj_exp))

    # Put all values in common data frame
    df_estimates_ee = merge_estimates(df_estimates_cont, df_estimates_exp)
    df_estimates_ee.name = "df_estimates_ee"
    safe_save_dataframe(df_estimates_ee)

    # Plot BIC
    plt.figure()
    plt.bar([0, 1, 2, 3], [df_estimates_cont[4], bic_rand_cont, df_estimates_exp[4], bic_rand_exp])
    plt.ylabel("BIC (higher better)")
    plt.xticks([0, 1, 2, 3],
               ['Control\nRescorla-Wagner', 'Control\nRandom', 'Experimental\nRescorla-Wagner', 'Experimental\nRandom'])

    # ------------------------
    # Out-of-sample prediction
    # ------------------------

    # Reset subject numbers for each group
    df_enrichment_exp, df_enrichment_cont = prepare_out_of_sample(df_enrichment_exp, df_enrichment_cont)

    # Extract even trials to predict odd trials in en_S_figure_6
    df_enrichment_exp_even = df_enrichment_exp[df_enrichment_exp['subj_num'] % 2 == 0]
    df_enrichment_cont_even = df_enrichment_cont[df_enrichment_cont['subj_num'] % 2 == 0]

    # Estimate free parameters
    df_estimates_cont_even = estimation_func(est_vars, agent_vars, df_enrichment_cont_even)
    df_estimates_exp_even = estimation_func(est_vars, agent_vars, df_enrichment_exp_even)

    # Put all values in common data frame
    df_estimates_ee_even = merge_estimates(df_estimates_cont_even, df_estimates_exp_even)
    df_estimates_ee_even.name = "df_estimates_ee_even"
    safe_save_dataframe(df_estimates_ee_even)

    # --------------------------------------
    #  4. Estimate model for Rac1 experiment
    # --------------------------------------

    # Estimate free parameters
    df_estimates_cont = estimation_func(est_vars, agent_vars, df_rac1_cont)
    df_estimates_exp = estimation_func(est_vars, agent_vars, df_rac1_exp)

    # Put all values in common data frame
    df_estimates_rac1 = merge_estimates(df_estimates_cont, df_estimates_exp)
    df_estimates_rac1.name = "df_estimates_rac1"
    safe_save_dataframe(df_estimates_rac1)

    # Print out summary of results
    print("Environmental Enrichment:")
    print(df_estimates_ee)
    print("\nRac1:")
    print(df_estimates_rac1)
    print("\nCross validation:")
    print(df_estimates_ee_even)

    plt.ioff()
    plt.show()
