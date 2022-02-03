import numpy as np
import pandas as pd
from math import inf

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe


def filter_b(the_df, b):
    return the_df[the_df['b'] == b].drop('b', axis=1)


def try_get_energy(bb, v, the_df, i):
    try:
        return min(bb * the_df.iloc[i]['best_energy_by_sample'], v**2)
    except:
        return v**2


# generate relative difference
def rel_diff(exact, experimental):
    exact = float(exact)
    experimental = float(experimental)
    if exact == experimental:
        r = 0
    elif experimental == inf:
        r = 1
    else:
        r = abs(exact - experimental) / max(abs(exact), abs(experimental))
    return r

# generate graphs
GRAPHS_PATH = "data/graphs.pickle"
graph_df = generate_graph_dataframe(GRAPHS_PATH)

# generate experiments
EXPERIMENTS_PATH = "data/experiments.pickle"
experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df)

# load approximated solutions
# CLASSICAL_HEURISTIC_PATH = "data_gedlibpy/approx_solutions_2.csv"
# heur_df = pd.read_csv(CLASSICAL_HEURISTIC_PATH)
#
# CLASSICAL_HEURISTIC_RD_PATH = "relative_differences/heur_rd.pickle"
# HEUR_ALGO_LIST = [
#     'BRANCH', 'BRANCH_FAST', 'BRANCH_TIGHT',
#     'BRANCH_UNIFORM', 'BRANCH_COMPACT', 'PARTITION', 'HYBRID',
#     'RING', 'ANCHOR_AWARE_GED', 'WALKS', 'IPFP', 'BIPARTITE',
#     'SUBGRAPH', 'NODE', 'RING_ML', 'BIPARTITE_ML', 'REFINE',
#     'BP_BEAM', 'SIMULATED_ANNEALING', 'HED', 'STAR'
# ]
# HEUR_RELATIVE_DIFFERENCE_COLS = ['experiment', 'v'] + HEUR_ALGO_LIST

# heur_rd_df = pd.DataFrame(columns=HEUR_RELATIVE_DIFFERENCE_COLS)
# for i in range(len(heur_df)):
#     heur_row = heur_df.iloc[i]
#     row = {'experiment': int(heur_row['experiment']), 'v': int(heur_row['v'])}
#     for algo in HEUR_ALGO_LIST:
#         exact = float(heur_row['exact_distance'])
#         experimental = float(eval(heur_row[algo])[0])
#         r = rel_diff(exact, experimental)
#         row[algo] = r
#     heur_rd_df.loc[len(heur_rd_df)] = row
# heur_rd_df.to_pickle(CLASSICAL_HEURISTIC_RD_PATH)

# heur_rd_df = pd.read_pickle(CLASSICAL_HEURISTIC_RD_PATH)
# heur_rd_results = heur_rd_df.drop('experiment', axis=1).fillna(1).groupby(['v']).mean()
#
#
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
# heur_rd_results.to_latex("relative_differences/heur_rd.tex")
#
# heur_succ_prob_df = (heur_rd_df < 0.01).astype(int)
# heur_succ_prob_df['v'] = heur_rd_df['v']
# heur_succ_prob_df = heur_succ_prob_df.drop('experiment', axis=1).fillna(1).groupby(['v']).mean()
# heur_succ_prob_df.to_latex("relative_differences/heur_success_prob.tex")
#
# heur_hq_prob_df = (heur_rd_df <= 0.20).astype(int)
# heur_hq_prob_df['v'] = heur_rd_df['v']
# heur_hq_prob_df = heur_hq_prob_df.drop('experiment', axis=1).fillna(1).groupby(['v']).mean()
# heur_hq_prob_df.to_latex("relative_differences/heur_hq_prob.tex")

# =====================================================================

sa_solutions_df             = pd.read_pickle("data/simulated_annealing_solutions.pickle")
dwave_2000_1us_solutions_df = pd.read_pickle("results_ric/2000_01_ann1micros.pickle")
dwave_2000_20us_solutions_df= pd.read_pickle("results_ric/2000_01_ann20micros.pickle")
dwave_2000_dc_solutions_df  = pd.read_pickle("data/dwave_2000_solutions.pickle")
dwave_2000_lt_solutions_df  = pd.read_pickle("data/dwave_2000_solutions_LP.pickle")
dwave_2000_pm_solutions_df  = pd.read_pickle("data/dwave_2000_solutions_PM.pickle")
dwave_adv_1us_solutions_df  = pd.read_pickle("results_ric/advantage_01_ann1micros.pickle")
dwave_adv_20us_solutions_df = pd.read_pickle("results_ric/advantage_01_ann20micros.pickle")
dwave_adv_dc_solutions_df   = pd.read_pickle("data/dwave_advantage_solutions.pickle")
dwave_adv_lt_solutions_df   = pd.read_pickle("data/dwave_advantage_solutions_LP.pickle")
dwave_adv_pm_solutions_df   = pd.read_pickle("data/dwave_advantage_solutions_PM.pickle")
dwave_leap_solutions_df     = pd.read_pickle("data/dwave_leap_solutions.pickle")
vqe_p1_solutions_df         = pd.read_pickle("data/vqe_p1_solutions.pickle")
vqe_p3_solutions_df         = pd.read_pickle("data/vqe_p3_solutions.pickle")
qaoa_p1_solutions_df        = pd.read_pickle("data/qaoa_p1_solutions.pickle")
qaoa_p3_solutions_df        = pd.read_pickle("data/qaoa_p3_solutions.pickle")

sa_solutions_df['V']             = experiments_df['vertices']
dwave_2000_dc_solutions_df['V']  = experiments_df['vertices']
dwave_2000_lt_solutions_df['V']  = experiments_df['vertices']
dwave_2000_pm_solutions_df['V']  = experiments_df['vertices']
dwave_adv_dc_solutions_df['V']   = experiments_df['vertices']
dwave_adv_lt_solutions_df['V']   = experiments_df['vertices']
dwave_adv_pm_solutions_df['V']   = experiments_df['vertices']
dwave_leap_solutions_df['V']     = experiments_df['vertices']
vqe_p1_solutions_df['V']         = experiments_df['vertices']
vqe_p3_solutions_df['V']         = experiments_df['vertices']
qaoa_p1_solutions_df['V']        = experiments_df['vertices']
qaoa_p3_solutions_df['V']        = experiments_df['vertices']

sa_solutions_df['b']             = experiments_df['b']
dwave_2000_dc_solutions_df['b']  = experiments_df['b']
dwave_2000_lt_solutions_df['b']  = experiments_df['b']
dwave_2000_pm_solutions_df['b']  = experiments_df['b']
dwave_adv_dc_solutions_df['b']   = experiments_df['b']
dwave_adv_lt_solutions_df['b']   = experiments_df['b']
dwave_adv_pm_solutions_df['b']   = experiments_df['b']
dwave_leap_solutions_df['b']     = experiments_df['b']
vqe_p1_solutions_df['b']         = experiments_df['b']
vqe_p3_solutions_df['b']         = experiments_df['b']
qaoa_p1_solutions_df['b']        = experiments_df['b']
qaoa_p3_solutions_df['b']        = experiments_df['b']

b = 0.10
experiments_df              = filter_b(experiments_df              , b)
sa_solutions_df             = filter_b(sa_solutions_df             , b)
dwave_2000_dc_solutions_df  = filter_b(dwave_2000_dc_solutions_df  , b)
dwave_2000_lt_solutions_df  = filter_b(dwave_2000_lt_solutions_df  , b)
dwave_2000_pm_solutions_df  = filter_b(dwave_2000_pm_solutions_df  , b)
dwave_adv_dc_solutions_df   = filter_b(dwave_adv_dc_solutions_df   , b)
dwave_adv_lt_solutions_df   = filter_b(dwave_adv_lt_solutions_df   , b)
dwave_adv_pm_solutions_df   = filter_b(dwave_adv_pm_solutions_df   , b)
dwave_leap_solutions_df     = filter_b(dwave_leap_solutions_df     , b)
vqe_p1_solutions_df         = filter_b(vqe_p1_solutions_df         , b)
vqe_p3_solutions_df         = filter_b(vqe_p3_solutions_df         , b)
qaoa_p1_solutions_df        = filter_b(qaoa_p1_solutions_df        , b)
qaoa_p3_solutions_df        = filter_b(qaoa_p3_solutions_df        , b)

quantum_rd_df = pd.DataFrame(columns=['experiment', 'V', 'SA', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'V1', 'V2', 'Q1', 'Q2'])
for i in range(len(experiments_df)):
    v = experiments_df.iloc[i]['vertices']
    row = {
        'experiment': i,
        'V': v,
        'SA': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, sa_solutions_df, i)),
        'D1': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_2000_dc_solutions_df, i)),
        'D2': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_2000_lt_solutions_df, i)),
        'D3': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_2000_pm_solutions_df, i)),
        'D4': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_adv_dc_solutions_df, i)),
        'D5': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_adv_lt_solutions_df, i)),
        'D6': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_adv_pm_solutions_df, i)),
        'D7': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, dwave_leap_solutions_df, i)),
        'V1': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, vqe_p1_solutions_df, i)),
        'V2': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, vqe_p3_solutions_df, i)),
        'Q1': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, qaoa_p1_solutions_df, i)),
        'Q2': rel_diff(experiments_df.iloc[i]['exact_distance'], try_get_energy(1/b, v, qaoa_p3_solutions_df, i))
    }
    quantum_rd_df.loc[len(quantum_rd_df)] = row

quantum_rd_df.to_pickle("relative_differences/quantum_rd_df.pickle")

quantum_rd_results = quantum_rd_df.drop('experiment', axis=1).fillna(1).groupby(['V']).mean()
pd.set_option('display.float_format', lambda x: '%.2f' % x)
quantum_rd_results.to_latex("relative_differences/quantum_rd_results.tex")

quantum_succ_prob_df = (quantum_rd_df < 0.01).astype(int)
quantum_succ_prob_df['v'] = quantum_rd_df['V']
quantum_succ_prob_df = quantum_succ_prob_df.drop('experiment', axis=1).fillna(1).groupby(['v']).mean()
quantum_succ_prob_df.to_latex("relative_differences/quantum_success_prob.tex")

quantum_hq_prob_df = (quantum_rd_df <= 0.20).astype(int)
quantum_hq_prob_df['v'] = quantum_rd_df['V']
quantum_hq_prob_df = quantum_hq_prob_df.drop('experiment', axis=1).fillna(1).groupby(['v']).mean()
quantum_hq_prob_df.to_latex("relative_differences/quantum_hq_prob.tex")
