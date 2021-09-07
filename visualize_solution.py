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
import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.options.display.float_format = "{:,.2f}".format


#PATHS
EXPERIMENTS_PATH = "data/experiments.pickle"
SIM_PATH = "data/simulated_annealing_solutions.pickle"
DWAVE_2000_PATH = "data/dwave_2000_solutions.pickle"
DWAVE_2000_1_PATH = "data/dwave_2000_solutions_LP.pickle"
DWAVE_2000_2_PATH = "data/dwave_2000_solutions_PM.pickle"
DWAVE_ADV_PATH = "data/dwave_advantage_solutions.pickle"
DWAVE_ADV_1_PATH = "data/dwave_advantage_solutions_LP.pickle"
DWAVE_ADV_2_PATH = "data/dwave_advantage_solutions_PM.pickle"
DWAVE_LEAP_PATH = "data/dwave_leap_solutions.pickle"

#DATAFRAMES
experiments_df = pd.read_pickle(EXPERIMENTS_PATH)
sim_df = pd.read_pickle(SIM_PATH)
dwave_2000_df = pd.read_pickle(DWAVE_2000_PATH)
dwave_2000_lp_df = pd.read_pickle(DWAVE_2000_1_PATH)
dwave_2000_pm_df = pd.read_pickle(DWAVE_2000_2_PATH)
dwave_adv_df = pd.read_pickle(DWAVE_ADV_PATH)
dwave_adv_lp_df = pd.read_pickle(DWAVE_ADV_1_PATH)
dwave_adv_pm_df = pd.read_pickle(DWAVE_ADV_2_PATH)
dwave_leap_df = pd.read_pickle(DWAVE_LEAP_PATH)

#Select columns
experiments_sel_df = experiments_df['exact_distance']
b_df = experiments_df['b']
sim_sel_df = sim_df['best_energy_by_sample']
dwave_2000_sel_df = dwave_2000_df['best_energy_by_sample']
dwave_2000_lp_sel_df = dwave_2000_lp_df['best_energy_by_sample']
dwave_2000_pm_sel_df = dwave_2000_pm_df['best_energy_by_sample']

padding_dwave_2000_sel_df = dwave_2000_sel_df
padding_dwave_2000_lp_sel_df = dwave_2000_lp_sel_df
padding_dwave_2000_pm_sel_df = dwave_2000_pm_sel_df

for i in range(len(dwave_2000_sel_df), len(experiments_sel_df)):
    padding_dwave_2000_sel_df.loc[len(padding_dwave_2000_sel_df)] = None
    padding_dwave_2000_lp_sel_df.loc[len(padding_dwave_2000_lp_sel_df)] = None
    padding_dwave_2000_pm_sel_df.loc[len(padding_dwave_2000_pm_sel_df)] = None

dwave_adv_sel_df = dwave_adv_df['best_energy_by_sample']
dwave_adv_lp_sel_df = dwave_adv_lp_df['best_energy_by_sample']
dwave_adv_pm_sel_df = dwave_adv_pm_df['best_energy_by_sample']
dwave_leap_sel_df = dwave_leap_df['best_energy_by_sample']

results_NOB_df = pd.read_pickle("data/ALL_best_energy_by_sample_NO_B.pickle")
results_WITHB_df = pd.read_pickle("data/ALL_best_energy_by_sample_WITH_B.pickle")

variables = [9, 16, 25, 36, 49, 64, 81]
vertices = [3, 4, 5, 6, 7, 8, 9]

diff_df = pd.read_pickle("data/difference_value_simple.pickle")

v_3 = diff_df[diff_df['vertices'] == 3].drop('vertices', axis=1)
v_4 = diff_df[diff_df['vertices'] == 4].drop('vertices', axis=1)
v_5 = diff_df[diff_df['vertices'] == 5].drop('vertices', axis=1)
v_6 = diff_df[diff_df['vertices'] == 6].drop('vertices', axis=1)
v_7 = diff_df[diff_df['vertices'] == 7].drop('vertices', axis=1)
v_8 = diff_df[diff_df['vertices'] == 8].drop('vertices', axis=1)
v_9 = diff_df[diff_df['vertices'] == 9].drop('vertices', axis=1)


v_3_b1 = v_3[v_3['b'] == 0.1].drop('b', axis=1)
v_3_b2 = v_3[v_3['b'] == 0.05].drop('b', axis=1)
v_3_b3 = v_3[v_3['b'] == 0.01].drop('b', axis=1)

v_4_b1 = v_4[v_4['b'] == 0.1].drop('b', axis=1)
v_4_b2 = v_4[v_4['b'] == 0.05].drop('b', axis=1)
v_4_b3 = v_4[v_4['b'] == 0.01].drop('b', axis=1)

v_5_b1 = v_5[v_5['b'] == 0.1].drop('b', axis=1)
v_5_b2 = v_5[v_5['b'] == 0.05].drop('b', axis=1)
v_5_b3 = v_5[v_5['b'] == 0.01].drop('b', axis=1)

v_6_b1 = v_6[v_6['b'] == 0.1].drop('b', axis=1)
v_6_b2 = v_6[v_6['b'] == 0.05].drop('b', axis=1)
v_6_b3 = v_6[v_6['b'] == 0.01].drop('b', axis=1)

v_7_b1 = v_7[v_7['b'] == 0.1].drop('b', axis=1)
v_7_b2 = v_7[v_7['b'] == 0.05].drop('b', axis=1)
v_7_b3 = v_7[v_7['b'] == 0.01].drop('b', axis=1)

v_8_b1 = v_8[v_8['b'] == 0.1].drop('b', axis=1)
v_8_b2 = v_8[v_8['b'] == 0.05].drop('b', axis=1)
v_8_b3 = v_8[v_8['b'] == 0.01].drop('b', axis=1)

v_9_b1 = v_9[v_9['b'] == 0.1].drop('b', axis=1)
v_9_b2 = v_9[v_9['b'] == 0.05].drop('b', axis=1)
v_9_b3 = v_9[v_9['b'] == 0.01].drop('b', axis=1)

diff_qaoa_vqe = pd.read_pickle("data/qaoa_vqe_difference_WITHB.pickle")
#q_p1 = pd.read_pickle("data/qaoa_p1_solutions.pickle")['best_energy_by_sample']
#q_p3 = pd.read_pickle("data/qaoa_p3_solutions.pickle")['best_energy_by_sample']
#vq_p3 = pd.read_pickle("data/vqe_p3_solutions.pickle")['best_energy_by_sample']
#vq_p1 = pd.read_pickle("data/vqe_p1_solutions.pickle")['best_energy_by_sample']
#
#columns = ['v','b','GED', 'GED_vs_qaoaP1', 'GED_vs_qaoaP3', 'GED_vs_vqeP1','GED_vs_vqeP3']
#new_diff = pd.DataFrame(columns = ['v','b','GED'])
#for i in range(len(q_p1)):
#    new_diff.loc[len(new_diff)] = {'v': diff_df['vertices'][i], 'b': b_df[i], 'GED': experiments_sel_df[i]}
#
##ged_qp1 = pd.DataFrame(columns = ['diff'])
#for i in range(len(q_p1)):
#    q_p1.iloc[i] = np.abs((q_p1.iloc[i]/b_df.iloc[i]) - experiments_sel_df.iloc[i])
#    q_p3.iloc[i] = np.abs((q_p3.iloc[i]/b_df.iloc[i]) - experiments_sel_df.iloc[i])
#    vq_p1.iloc[i] = np.abs((vq_p1.iloc[i]/b_df.iloc[i]) - experiments_sel_df.iloc[i])
#    vq_p3.iloc[i] = np.abs((q_p3.iloc[i]/b_df.iloc[i]) - experiments_sel_df.iloc[i])
#
#new_diff_df = pd.concat([new_diff['v'], new_diff['b'], new_diff['GED'], q_p1, q_p3,vq_p1,vq_p3], axis=1)
#new_diff_df.columns = columns
#


#
#columns = ['vertices','simM','simS', '2000M','2000S', '2000ltM', '2000ltS', '2000pmM', '2000pmS', 'advM', 'advS', 'advltM', 'advltS', 'advpmM', 'advpmS', 'leapM', 'leapS']
#mean_var_df = pd.DataFrame(columns = columns)
#
#for v in vertices:
#    row = {'vertices': v,'simM':diff_df[diff_df['vertices'] == v]['GED_vs_sim'].mean(),'simS':diff_df[diff_df['vertices'] == v]['GED_vs_sim'].std(),
#           '2000M':diff_df[diff_df['vertices'] == v]['GED_vs_2000'].mean(),'2000S':diff_df[diff_df['vertices'] == v]['GED_vs_2000'].std(),
#           '2000ltM':diff_df[diff_df['vertices'] == v]['GED_vs_2000_lt'].mean(), '2000ltS':diff_df[diff_df['vertices'] == v]['GED_vs_2000_lt'].std(),
#           '2000pmM':diff_df[diff_df['vertices'] == v]['GED_vs_2000_pm'].mean(), '2000pmS':diff_df[diff_df['vertices'] == v]['GED_vs_2000_pm'].std(),
#           'advM':diff_df[diff_df['vertices'] == v]['GED_vs_adv'].mean(), 'advS':diff_df[diff_df['vertices'] == v]['GED_vs_adv'].std(),
#           'advltM':diff_df[diff_df['vertices'] == v]['GED_vs_adv_lt'].mean(), 'advltS':diff_df[diff_df['vertices'] == v]['GED_vs_adv_lt'].std(),
#           'advpmM':diff_df[diff_df['vertices'] == v]['GED_vs_adv_pm'].mean(), 'advpmS':diff_df[diff_df['vertices'] == v]['GED_vs_adv_pm'].std(),
#           'leapM':diff_df[diff_df['vertices'] == v]['GED_vs_leap'].mean(), 'leapS':diff_df[diff_df['vertices'] == v]['GED_vs_leap'].std()}
#    mean_var_df.loc[len(mean_var_df)] = row
#
#diff_df = pd.concat([experiments_df['vertices'], experiments_df['b'], diff_ged_sim, diff_ged_2000, diff_ged_2000_lp, diff_ged_2000_pm, diff_ged_adv, diff_ged_adv_lp, diff_ged_adv_pm, diff_ged_leap], axis=1)#results_df = pd.concat([experiments_df['vertices'], experiments_sel_df, sim_sel_df, padding_dwave_2000_sel_df, padding_dwave_2000_lp_sel_df, padding_dwave_2000_pm_sel_df, dwave_adv_sel_df, dwave_adv_lp_sel_df, dwave_adv_pm_sel_df, dwave_leap_sel_df], axis = 1)
#df_columns = ['vertices','GED', 'BEBS_sim', 'BEBS_2000', 'BEBS_2000_lt', 'BEBS_2000_pm', 'BEBS_adv', 'BEBS_adv_lt', 'BEBS_adv_pm', 'BEBS_leap']
#results_df.columns = df_columns
##print(results_df)
#
#for i in range(len(experiments_sel_df)):
#    sim_sel_df.iloc[i] = sim_sel_df.iloc[i]/b_df.iloc[i]
#    if padding_dwave_2000_sel_df.iloc[i] is not None:
#        padding_dwave_2000_sel_df.iloc[i] = padding_dwave_2000_sel_df.iloc[i]/b_df.iloc[i]
#        padding_dwave_2000_lp_sel_df.iloc[i] = padding_dwave_2000_lp_sel_df.iloc[i]/b_df.iloc[i]
#        padding_dwave_2000_pm_sel_df.iloc[i] = padding_dwave_2000_pm_sel_df.iloc[i]/b_df.iloc[i]
#    dwave_adv_sel_df.iloc[i] = dwave_adv_sel_df.iloc[i]/b_df.iloc[i]
#    dwave_adv_lp_sel_df.iloc[i] = dwave_adv_lp_sel_df.iloc[i]/b_df.iloc[i]
#    dwave_adv_pm_sel_df.iloc[i] = dwave_adv_pm_sel_df.iloc[i]/b_df.iloc[i]
#    dwave_leap_sel_df.iloc[i] = dwave_leap_sel_df.iloc[i]/b_df.iloc[i]
#
#results_withB_df = pd.concat([experiments_df['vertices'], experiments_sel_df, sim_sel_df, padding_dwave_2000_sel_df, padding_dwave_2000_lp_sel_df, padding_dwave_2000_pm_sel_df, dwave_adv_sel_df, dwave_adv_lp_sel_df, dwave_adv_pm_sel_df, dwave_leap_sel_df],axis = 1)
#results_withB_df.columns = df_columns
#
#results_NOB_df = pd.read_pickle("data/ALL_best_energy_by_sample_NO_B.pickle")
#results_WITHB_df = pd.read_pickle("data/ALL_best_energy_by_sample_WITH_B.pickle")
#
#variables = [9, 16, 25, 36, 49, 64, 81]
#vertices = [3, 4, 5, 6, 7, 8, 9]
#
#diff_df = pd.read_pickle("data/difference_value_simple.pickle")
#
#CORRETTO:
#
#diff_ged_sim = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_sim.loc[len(diff_ged_sim)] = np.abs(results_WITHB_df['BEBS_sim'][i] - experiments_sel_df.iloc[i])
#
#diff_ged_2000 = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    if results_WITHB_df['BEBS_2000'][i] is None:
#        diff_ged_2000.loc[len(diff_ged_2000)] = {}
#    else:
#        diff_ged_2000.loc[len(diff_ged_2000)] = np.abs(results_WITHB_df['BEBS_2000'][i] - experiments_sel_df.iloc[i])
#
#
#diff_ged_2000_lp = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    if results_WITHB_df['BEBS_2000_lt'][i] is None:
#        diff_ged_2000_lp.loc[len(diff_ged_2000_lp)] = {}
#    else:
#        diff_ged_2000_lp.loc[len(diff_ged_2000_lp)] = np.abs(results_WITHB_df['BEBS_2000_lt'][i] - experiments_sel_df.iloc[i])
#
#diff_ged_2000_pm = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    if results_WITHB_df['BEBS_2000_pm'][i] is None:
#        diff_ged_2000_pm.loc[len(diff_ged_2000_pm)] = {}
#    else:
#        diff_ged_2000_pm.loc[len(diff_ged_2000_pm)] = np.abs(results_WITHB_df['BEBS_2000_pm'][i] - experiments_sel_df.iloc[i])
#
#diff_ged_adv = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_adv.loc[len(diff_ged_adv)] = np.abs(results_WITHB_df['BEBS_adv'][i] - experiments_sel_df.iloc[i])
#
#diff_ged_adv_lp = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_adv_lp.loc[len(diff_ged_adv_lp)] = np.abs(results_WITHB_df['BEBS_adv_lt'][i] - experiments_sel_df.iloc[i])
#
#diff_ged_adv_pm = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_adv_pm.loc[len(diff_ged_adv_pm)] = np.abs(results_WITHB_df['BEBS_adv_pm'][i] - experiments_sel_df.iloc[i])
#
#diff_ged_leap = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_leap.loc[len(diff_ged_leap)] = np.abs(results_WITHB_df['BEBS_leap'][i] - experiments_sel_df.iloc[i])
#
#
#
#columns = ['vertices', 'b','GED_vs_sim', 'GED_vs_2000', 'GED_vs_2000_lt', 'GED_vs_2000_pm', 'GED_vs_adv', 'GED_vs_adv_lt', 'GED_vs_adv_pm', 'GED_vs_leap']
#diff_df = pd.concat([experiments_df['vertices'], experiments_df['b'], diff_ged_sim, diff_ged_2000, diff_ged_2000_lp, diff_ged_2000_pm, diff_ged_adv, diff_ged_adv_lp, diff_ged_adv_pm, diff_ged_leap], axis=1)
#diff_df.columns = columns

#diff_ged_sim = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_sim.loc[len(diff_ged_sim)] = np.abs(sim_sel_df.iloc[i] - experiments_sel_df.iloc[i])
#
#diff_ged_2000 = pd.DataFrame(columns=['difference'])
#for i in range(270):
#    diff_ged_2000.loc[len(diff_ged_2000)] = np.abs(dwave_2000_sel_df.iloc[i] - experiments_sel_df.iloc[i])
#
#diff_ged_adv = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_adv.loc[len(diff_ged_adv)] = np.abs(dwave_adv_sel_df.iloc[i] - experiments_sel_df.iloc[i])
#
#diff_ged_leap = pd.DataFrame(columns=['difference'])
#for i in range(len(experiments_df)):
#    diff_ged_leap.loc[len(diff_ged_leap)] = np.abs(dwave_leap_sel_df.iloc[i] - experiments_sel_df.iloc[i])
#
#columns = ['GED_vs_sim', 'GED_vs_2000', 'GED_vs_adv', 'GED_vs_leap', 'b']
#diff_df = pd.concat([diff_ged_sim, diff_ged_2000, diff_ged_adv, diff_ged_leap, experiments_df['b']], axis=1)
#diff_df.columns = columns
#
#print(results_withB_df)
