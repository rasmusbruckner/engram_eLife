""" Engram Utilities: This module contains utility functions that are repeatedly used throughout the project """

import numpy as np
import pandas as pd
import os


def safe_save_dataframe(dataframe):
    """ This function saves a data frame and ensures that values don't change unexpectedly

    :param dataframe: Data frame that we want to save
    :return: None
    """

    # Initialize expected data frame
    expected_df = np.nan

    # File path
    df_name = 'en_data/' + dataframe.name + '.pkl'

    # Check if file exists
    path_exist = os.path.exists(df_name)

    # If so, load file for comparison
    if path_exist:
        expected_df = pd.read_pickle(df_name)

    # If we have the file already, check if as expected
    if path_exist:
        # Test if equal and save data
        same = dataframe.equals(expected_df)
        print("\nActual and expected " + dataframe.name + " equal:", same, "\n")

    # If new, we'll create the file
    else:
        same = True
        print("\nCreating new data frame: " + dataframe.name + "\n")

    if not same:
        dataframe.to_pickle('en_data/' + dataframe.name + '_unexpected.pkl')
    else:
        dataframe.to_pickle('en_data/' + dataframe.name + '.pkl')


def get_group_data(path):
    """ This function loads the required data set and returns group data

    :param path: Path to folder with raw data
    :return: df_cont: Control group
    :return: df_exp: Experimental group
    """

    # Load data
    df_data = pd.read_pickle(path)

    # Control group
    df_cont = df_data[df_data["group"] == 0]
    df_cont = df_cont.reset_index()
    df_cont["subj_num"] = df_cont["subj_num"] - min(df_cont["subj_num"])

    # Experimental group
    df_exp = df_data[df_data["group"] == 1]
    df_exp = df_exp.reset_index()
    df_exp["subj_num"] = df_exp["subj_num"] - min(df_exp["subj_num"])

    return df_cont, df_exp


def merge_estimates(df_estimates_cont, df_estimates_exp):
    """ This function puts the estimates into a common data frame

    :param df_estimates_cont: Parameters of control group
    :param df_estimates_exp: Parameters of experimental group
    :return df estimate: Common data frame with parameters

    """

    # Create data frame
    df_estimates = pd.DataFrame(np.nan, index=[0, 1], columns=['kappa', 'beta', 'alpha'])

    # Add parameter estimates
    df_estimates.loc[0, "kappa"] = df_estimates_cont[0]
    df_estimates.loc[0, "beta"] = df_estimates_cont[1]
    df_estimates.loc[0, "alpha"] = df_estimates_cont[2]
    df_estimates.loc[0, "ll"] = df_estimates_cont[3]
    df_estimates.loc[0, "BIC"] = df_estimates_cont[4]
    df_estimates.loc[1, "kappa"] = df_estimates_exp[0]
    df_estimates.loc[1, "beta"] = df_estimates_exp[1]
    df_estimates.loc[1, "alpha"] = df_estimates_exp[2]
    df_estimates.loc[1, "ll"] = df_estimates_exp[3]
    df_estimates.loc[1, "BIC"] = df_estimates_exp[4]

    return df_estimates


def prepare_out_of_sample(df_enrichment_exp, df_enrichment_cont):
    """ This function resets subject number for each group (cont, exp) for out-of-sample analyses

    :param df_enrichment_exp: Data frame experimental group
    :param df_enrichment_cont: Data frame control group
    :return: df_enrichment_exp:  Updated data frame experimental group
    :return: df_enrichment_cont: Updated data frame control group

    """

    # Number of subjects
    n_subj_exp = len(list(set(df_enrichment_exp['subj_num'])))
    n_subj_cont = len(list(set(df_enrichment_cont['subj_num'])))

    # Reset index of data frame
    df_enrichment_exp = df_enrichment_exp.reset_index(drop=True)
    df_enrichment_cont = df_enrichment_cont.reset_index(drop=True)

    # Replace actual ID with subject number within current group (therefore not done in preprocessing)
    df_enrichment_exp['subj_num'] = np.repeat(np.arange(n_subj_exp), 5)
    df_enrichment_cont['subj_num'] = np.repeat(np.arange(n_subj_cont), 5)

    return df_enrichment_exp, df_enrichment_cont


def safe_div(x, y):
    """ This function divides two numbers and avoids division by zero

        Obtained from:
        https://www.yawintutor.com/zerodivisionerror-division-by-zero/

    :param x: x-value
    :param y: y-value
    :return: c: result
    """

    if y == 0:
        c = 0.0
    else:
        c = x / y
    return c
