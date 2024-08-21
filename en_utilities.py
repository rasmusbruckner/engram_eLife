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
