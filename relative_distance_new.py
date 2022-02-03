import numpy as np
import pandas as pd
from math import inf
import matplotlib.pyplot as plt

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
from dwavesolutioncreator import generate_hamiltonian


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


def normalize_cost(g1, g2, v, sample):
    if g1 is None or g2 is None or v is None or sample is None:
        return None if v is None else v**2

    HARD_CONSTRAINT = 10000
    model = generate_hamiltonian(g1, g2, HARD_CONSTRAINT, 1).compile()
    e = model.to_bqm().energy(sample)
    return min(e, v**2)


# generate graphs
GRAPHS_PATH = "data/graphs.pickle"
graph_df = generate_graph_dataframe(GRAPHS_PATH)

# generate experiments
EXPERIMENTS_PATH = "data/experiments.pickle"
experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df)

EXPERIMENTS_NEW = "results_ric/experiments_01_ann1micros.pickle"
experiments_new_df = generate_experiments_dataframe(EXPERIMENTS_NEW, graph_df)

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

OLD_DF = [sa_solutions_df, dwave_2000_dc_solutions_df, dwave_2000_lt_solutions_df, dwave_2000_pm_solutions_df,
          dwave_adv_dc_solutions_df, dwave_adv_lt_solutions_df, dwave_adv_pm_solutions_df, dwave_leap_solutions_df,
          vqe_p1_solutions_df, vqe_p3_solutions_df, qaoa_p1_solutions_df, qaoa_p3_solutions_df]
NEW_DF = [dwave_2000_1us_solutions_df, dwave_2000_20us_solutions_df, dwave_adv_1us_solutions_df, dwave_adv_20us_solutions_df]

for i, df in enumerate(OLD_DF):
    OLD_DF[i]['b'] = experiments_df['b']
    OLD_DF[i]['v'] = experiments_df['vertices']
    OLD_DF[i]['g1'] = experiments_df['g1']
    OLD_DF[i]['g2'] = experiments_df['g2']
    OLD_DF[i]['exact_distance'] = experiments_df['exact_distance']

for i, df in enumerate(OLD_DF):
    OLD_DF[i] = filter_b(OLD_DF[i], 0.10).reset_index()

for i in range(len(OLD_DF)):
    OLD_DF[i]['normalized_cost'] = OLD_DF[i].apply(lambda row: normalize_cost(row.g1, row.g2, row.v, row.best_sample), axis=1)
    OLD_DF[i]['relative_distance'] = OLD_DF[i].apply(lambda row: rel_diff(row.exact_distance, row.normalized_cost), axis=1)

for i, df in enumerate(NEW_DF):
    NEW_DF[i]['v'] = experiments_new_df['vertices']
    NEW_DF[i]['g1'] = experiments_new_df['g1']
    NEW_DF[i]['g2'] = experiments_new_df['g2']
    NEW_DF[i]['exact_distance'] = experiments_new_df['exact_distance']
    NEW_DF[i]['normalized_cost'] = NEW_DF[i].apply(lambda row: normalize_cost(row.g1, row.g2, row.v, row.best_sample), axis=1)
    NEW_DF[i]['relative_distance'] = NEW_DF[i].apply(lambda row: rel_diff(row.exact_distance, row.normalized_cost), axis=1)

for i, df in enumerate(OLD_DF+NEW_DF):
    print(f"Length of {i}th DF: {len(df)}")

relative_distance_df = pd.DataFrame()
relative_distance_df['v'] = experiments_new_df['vertices']

ORDERED_DF = [OLD_DF[0]] + NEW_DF[0:2] + OLD_DF[1:4] + NEW_DF[2:] + OLD_DF[4:]
NAMES_DF = ['SA',
            '2000 1us', '2000 20us', '2000 dc', '2000 lt', '2000 pm',
            'adv 1us', 'adv 20us', 'adv dc', 'adv lt', 'adv pm',
            'leap', 'vqe p1', 'vqe p3', 'qaoa p1', 'qaoa p3']

for i, (df, name) in enumerate(zip(ORDERED_DF, NAMES_DF)):
    print('processing ', name)
    relative_distance_df[name] = df['relative_distance']

relative_distance_df = relative_distance_df.fillna(1.0)

print("\n\nRELATIVE DISTANCE")
print(relative_distance_df.groupby(['v']).mean().round(decimals=2))

print("\n\nSUCCESS PROBABILITY")
success_probability = (relative_distance_df < 0.001).astype(int)
success_probability['v'] = relative_distance_df['v']
print(success_probability.groupby(['v']).mean().round(decimals=2))

print("\n\nHIGH QUALITY PROBABILITY")
hq_probability = (relative_distance_df <= 0.20).astype(int)
hq_probability['v'] = relative_distance_df['v']
print(hq_probability.groupby(['v']).mean().round(decimals=2))


# ===========================================

def create_plot(x, df, columns, labels, xlbl, ylbl, xlim=9.5):
    plt.figure()
    for i, (col, lbl) in enumerate(zip(columns, labels)):
        plt.plot(x, df[col][:len(x)], label=lbl)
    plt.xlabel(xlbl, fontsize=20)
    plt.ylabel(ylbl, fontsize=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((3, xlim))
    plt.ylim((0, 1.1))

x = range(3, 10)
rdg = relative_distance_df.groupby(['v']).mean().round(decimals=2)
spg = success_probability.groupby(['v']).mean().round(decimals=2)
hpg = hq_probability.groupby(['v']).mean().round(decimals=2)

# DWAVE plots configurations
create_plot(x, rdg, ['SA', '2000 1us', '2000 20us', '2000 dc', '2000 lt', '2000 pm'],
            ['SA', 'D1', 'D20', 'D21', 'D500', 'D100'], "N", "Average relative difference")
plt.savefig("plot_performances_new/2000_configurations_reldiff.png", dpi=300, bbox_inches='tight')
create_plot(x, rdg, ['SA', 'adv 1us', 'adv 20us', 'adv dc', 'adv lt', 'adv pm'],
            ['SA', 'D1', 'D20', 'D21', 'D500', 'D100'], "N", "Average relative difference")
plt.savefig("plot_performances_new/adv_configurations_reldiff.png", dpi=300, bbox_inches='tight')

create_plot(x, spg, ['SA', '2000 1us', '2000 20us', '2000 dc', '2000 lt', '2000 pm'],
            ['SA', 'D1', 'D20', 'D21', 'D500', 'D100'], "N", "Success probability")
plt.savefig("plot_performances_new/2000_configurations_sp.png", dpi=300, bbox_inches='tight')
create_plot(x, spg, ['SA', 'adv 1us', 'adv 20us', 'adv dc', 'adv lt', 'adv pm'],
            ['SA', 'D1', 'D20', 'D21', 'D500', 'D100'], "N", "Success probability")
plt.savefig("plot_performances_new/adv_configurations_sp.png", dpi=300, bbox_inches='tight')

create_plot(x, hpg, ['SA', '2000 1us', '2000 20us', '2000 dc', '2000 lt', '2000 pm'],
            ['SA', 'D1', 'D20', 'D21', 'D500', 'D100'], "N", "High quality probability")
plt.savefig("plot_performances_new/2000_configurations_hqp.png", dpi=300, bbox_inches='tight')
create_plot(x, hpg, ['SA', 'adv 1us', 'adv 20us', 'adv dc', 'adv lt', 'adv pm'],
            ['SA', 'D1', 'D20', 'D21', 'D500', 'D100'], "N", "High quality probability")
plt.savefig("plot_performances_new/adv_configurations_hqp.png", dpi=300, bbox_inches='tight')

# DWAVE plots configurations
create_plot(x, rdg, ['SA', '2000 20us', 'adv 20us', 'leap'],
            ['SA', '2000', 'ADV', 'LEAP'], "N", "Average relative difference")
plt.savefig("plot_performances_new/dwave_configurations_reldiff.png", dpi=300, bbox_inches='tight')

create_plot(x, spg, ['SA', '2000 20us', 'adv 20us', 'leap'],
            ['SA', '2000', 'ADV', 'LEAP'], "N", "Success probability")
plt.savefig("plot_performances_new/dwave_configurations_sp.png", dpi=300, bbox_inches='tight')

create_plot(x, hpg, ['SA', '2000 20us', 'adv 20us', 'leap'],
            ['SA', '2000', 'ADV', 'LEAP'], "N", "High quality probability")
plt.savefig("plot_performances_new/dwave_configurations_hqp.png", dpi=300, bbox_inches='tight')

# variationals
x = range(3, 6)
create_plot(x, rdg, ['SA', 'vqe p1', 'vqe p3', 'qaoa p1', 'qaoa p3'],
            ['SA', 'VQEp1', 'VQEp3', 'QAOAp1', 'QAOAp3'], "N", "Average relative difference")
plt.savefig("plot_performances_new/variational_configurations_reldiff.png", dpi=300, bbox_inches='tight')

create_plot(x, spg, ['SA', 'vqe p1', 'vqe p3', 'qaoa p1', 'qaoa p3'],
            ['SA', 'VQEp1', 'VQEp3', 'QAOAp1', 'QAOAp3'], "N", "Success probability")
plt.savefig("plot_performances_new/variational_configurations_sp.png", dpi=300, bbox_inches='tight')

create_plot(x, hpg, ['SA', 'vqe p1', 'vqe p3', 'qaoa p1', 'qaoa p3'],
            ['SA', 'VQEp1', 'VQEp3', 'QAOAp1', 'QAOAp3'], "N", "High quality probability")
plt.savefig("plot_performances_new/variational_configurations_hqp.png", dpi=300, bbox_inches='tight')

from qiskit.providers.aer.backends.backend_utils import (cpp_execute, available_methods, MAX_QUBITS_STATEVECTOR)
from qiskit.providers.aer.backends.controller_wrappers import qasm_controller_execute