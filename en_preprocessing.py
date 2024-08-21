""" Preprocessing: This script imports the behavioral data """

import numpy as np
import pandas as pd
from en_utilities import safe_save_dataframe


# Preprocessing of empirical data from O'Leary et al.
# --------------------------------------------------

def preprocess_data(df_raw):
    """ Preprocessing of the raw data

    :param df_raw: Raw data of experimental or control condition
    :return: df_prep: Preprocessed data
    """

    df_raw.columns = df_raw.iloc[0]
    df_prep = df_raw.drop([0, ])
    df_prep = df_prep.drop(columns=['Discrimination index'])
    df_prep = df_prep.reset_index(drop=True)
    df_prep.insert(0, 'subj_num', np.arange(len(df_prep)), True)
    df_prep = df_prep.rename(
        columns={"Animal ID": "ID", "Group": "group", "Treatment": "treatment", "Object 1": "acq_obj_1",
                 "Object 2": "acq_obj_2", "Familiar (sec)": "test_fam", "Novel (sec) ": "test_novel"})
    df_prep = df_prep.replace(to_replace=r'24 hr', value=0)  # name in enrichment experiment
    df_prep = df_prep.replace(to_replace=r'24hr', value=0)  # name in Rac1 experiment
    df_prep = df_prep.replace(to_replace=r'1 week', value=1)  # name in enrichment experiment
    df_prep = df_prep.replace(to_replace=r'1 week ITI', value=1)  # name in empty box experiment
    df_prep = df_prep.replace(to_replace=r'2 weeks', value=2)
    df_prep = df_prep.replace(to_replace=r'3 weeks', value=3)
    df_prep = df_prep.replace(to_replace=r'Control', value=0)  # control group enrichment, Rac1, and empty box
    df_prep = df_prep.replace(to_replace=r'Enrichment', value=1)  # enrichment group
    df_prep = df_prep.replace(to_replace=r'Ehop016', value=1)  # Rac1 group
    df_prep = df_prep.replace(to_replace=r'Empty Box', value=1)  # empty box group
    df_prep = df_prep.melt(id_vars=['subj_num', 'ID', 'group', 'treatment'])
    df_prep = df_prep.rename(columns={0: "object", "value": "exploration"})
    df_prep = df_prep.sort_values(by=['subj_num', 'object'])
    df_prep = df_prep.reset_index(drop=True)

    return df_prep


# Preprocessing of environmental enrichment experiment
# ----------------------------------------------------

# Load data
data_exp = pd.read_excel('en_data/en_data.xlsx', sheet_name='Forgetting Curve & Enrichment')

# Run preprocessing
df_main = preprocess_data(data_exp)

# Dropping this because of nan values
df_main.drop(df_main.index[df_main['subj_num'] == 57], inplace=True)

# Safe save
df_main.name = "en_prepr_data_ee"
safe_save_dataframe(df_main)

# Preprocessing of Rac1 experiment
# --------------------------------

# Load data
data_exp = pd.read_excel('en_data/en_data.xlsx', sheet_name='Rac1 Inhibition')

# Run preprocessing
df_main = preprocess_data(data_exp)

# Safe save
df_main.name = "en_prepr_data_rac1"
safe_save_dataframe(df_main)

# Preprocessing of empty box experiment
# -------------------------------------

# Load data
data_exp = pd.read_excel('en_data/en_data.xlsx', sheet_name='Empty Box Wild Type Behaviour')

# Run preprocessing
df_main = preprocess_data(data_exp)

# Safe save
df_main.name = "en_prepr_data_empty_box"
safe_save_dataframe(df_main)
