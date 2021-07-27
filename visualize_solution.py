import warnings
warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
from dwavesolutioncreator import generate_dwave_dataframe, run_dwave_2000_experiment, run_dwave_advantage_experiment, run_simulated_experiment, run_dwave_leap_experiment
#from qaoasolutioncreator import generate_qaoa_dataframe, run_qaoa_p1, run_qaoa_p3, run_qaoa_p5
import multiprocessing
import numpy as np
from datetime import datetime
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


#PATHS
EXPERIMENTS_PATH = "data/experiments.pickle"
SIM_PATH = "data/simulated_annealing_solutions.pickle"
DWAVE_2000_PATH = "data/dwave_2000_solutions.pickle"
DWAVE_ADVANTAGE_PATH = "data/dwave_advantage_solutions.pickle"
DWAVE_LEAP_PATH = "data/dwave_leap_solutions.pickle"

#DATAFRAMES
experiments_df = pd.read_pickle(EXPERIMENTS_PATH)
sim_df = pd.read_pickle(SIM_PATH)
dwave_2000_df = pd.read_pickle(DWAVE_2000_PATH)
dwave_adv_df = pd.read_pickle(DWAVE_ADVANTAGE_PATH)
dwave_leap_df = pd.read_pickle(DWAVE_LEAP_PATH)

#Select columns
experiments_sel_df = experiments_df['exact_distance']
b_df = experiments_df['b']
sim_sel_df = sim_df['best_energy_by_sample']
dwave_2000_sel_df = dwave_2000_df['best_energy_by_sample']
padding_dwave_2000_sel_df = dwave_2000_sel_df
dwave_adv_sel_df = dwave_adv_df['best_energy_by_sample']
dwave_leap_sel_df = dwave_leap_df['best_energy_by_sample']

print(b_df)
#for i in range(len(experiments_sel_df)):
#   print(experiments_sel_df.iloc[i], sim_sel_df.iloc[i], dwave_2000_sel_df.iloc[i], dwave_adv_sel_df.iloc[i], dwave_leap_sel_df.iloc[i])

for i in range(len(dwave_2000_sel_df), len(experiments_sel_df)):
    padding_dwave_2000_sel_df.loc[len(padding_dwave_2000_sel_df)] = None
results_df = pd.concat([experiments_sel_df, sim_sel_df, padding_dwave_2000_sel_df, dwave_adv_sel_df, dwave_leap_sel_df],axis = 1)
print(results_df)
