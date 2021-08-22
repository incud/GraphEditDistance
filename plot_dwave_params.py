import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.options.display.float_format = "{:,.2f}".format

experiments_df = pd.read_pickle("data/experiments.pickle")

dwave_2000_df    = pd.read_pickle("data/dwave_2000_solutions.pickle")
dwave_2000_lt_df = pd.read_pickle("data/dwave_2000_solutions_LP.pickle")
dwave_2000_pm_df = pd.read_pickle("data/dwave_2000_solutions_PM.pickle")

dwave_adv_df    = pd.read_pickle("data/dwave_advantage_solutions.pickle")
dwave_adv_lt_df = pd.read_pickle("data/dwave_advantage_solutions_LP.pickle")
dwave_adv_pm_df = pd.read_pickle("data/dwave_advantage_solutions_PM.pickle")

def get_stats(dwave_df, exp_df):

    dwave_df['vertices'] = exp_df['vertices']
    dwave_stats_df = dwave_df.drop(['experiment', 'start', 'end', 'best_sample',
                    'best_energy', 'best_energy_by_sample',
                    'best_sample_pp', 'best_energy_by_sample_pp', 'best_energy_pp'], axis=1
                ).fillna(0).groupby('vertices').mean()
    return dwave_stats_df
