import warnings

import pandas as pd
import neal
import numpy as np
from math import inf

warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
from dwavesolutioncreator import generate_dwave_dataframe
from experimentcreator import generate_hamiltonian
from greedy import SteepestDescentSolver

def calculate_relative_distance(sol_df, exp_df):
    return (sol_df['normalized_cost'] - exp_df['exact_distance']) / np.where(sol_df['normalized_cost'] > 0, sol_df['normalized_cost'], 1)


def check_cost(i):
    experiment = experiments_df.loc[i]
    sample = sim_df.loc[i]['best_sample']
    model = generate_hamiltonian(experiment["g1"], experiment["g2"], 10000, 0.001).compile()
    print(i, "-->", experiment["vertices"], sample)
    bqm = model.to_bqm()
    e = bqm.energy(sample)
    return e


def normalize_cost(experiment, sample):
    HARD_CONSTRAINT = 10000
    model = generate_hamiltonian(experiment["g1"], experiment["g2"], HARD_CONSTRAINT, 1).compile()
    e = model.to_bqm().energy(sample)
    return min(e, experiment["vertices"]**2)


def run_sa(i, experiment):

    print("Run SA")

    bqm = experiment["bqm"]
    model = generate_hamiltonian(experiment["g1"], experiment["g2"], experiment["a"], experiment["b"]).compile()

    sampler = neal.SimulatedAnnealingSampler()
    sample_set = sampler.sample(bqm, num_reads=10000, answer_mode='raw', return_embedding=True)
    decoded_samples = model.decode_sampleset(sample_set)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    # =========================================================================
    # Post-processing: see https://docs.ocean.dwavesys.com/en/stable/examples/pp_greedy.html#pp-greedy
    solver_greedy = SteepestDescentSolver()
    sample_set_pp = solver_greedy.sample(bqm, initial_states=sample_set)
    decoded_samples_pp = model.decode_sampleset(sample_set_pp)
    best_sample_pp = min(decoded_samples_pp, key=lambda x: x.energy)

    if experiment['vertices']**2 != len(best_sample.sample.keys()):
        print(i, "--> ERROR WHILE CALCULATIING SAMPLE, INCORRECT NUMBER OF VARIABLES")

    row = {
        "experiment": i,
        "start": 0,
        "end": 0,
        "l": len(best_sample.sample.keys()),
        "best_sample": best_sample.sample,
        "best_energy": best_sample.energy,
        "best_energy_by_sample": bqm.energy(best_sample.sample),
        "best_sample_pp": best_sample_pp.sample,
        "best_energy_pp": best_sample_pp.energy,
        "best_energy_by_sample_pp": bqm.energy(best_sample_pp.sample),
        "num_source_variables": 0,
        "num_target_variables": 0,
        "max_chain_length": 0,
        "chain_strength": 0,
        "chain_break_method": 0
    }
    return row


GRAPHS_PATH = "data/graphs.pickle"
graph_df = generate_graph_dataframe(GRAPHS_PATH)
graph_df.to_pickle(GRAPHS_PATH)

EXPERIMENTS_PATH = "data_estimate_b/experiments_estimate_b.pickle"
bs = [(1, 1/i) for i in range(1, 11)] + [(1, 0.05), (1, 0.01)]
experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df, bs)
experiments_df.to_pickle(EXPERIMENTS_PATH)

SIM_PATH = "data_estimate_b/sa_estimate_b.pickle"
sim_df = generate_dwave_dataframe(SIM_PATH)
sim_df.to_pickle(SIM_PATH)

# for i in range(len(experiments_df)):
#
#     print("i=", i)
#     the_experiment = experiments_df.loc[i]
#
#     print(f"running experiments {i} ...", end="")
#     if i < len(sim_df):
#         print("experiment already present")
#         continue
#
#     sim_df.loc[i] = run_sa(i, the_experiment)
#
#     if i % 20 == 0:
#         sim_df.to_pickle(SIM_PATH)

normalized_cost = np.array([0] * len(experiments_df))
for i in range(len(experiments_df)):
    normalized_cost[i] = normalize_cost(experiments_df.loc[i], sim_df.loc[i]['best_sample'])
sim_df['v'] = experiments_df['vertices']
sim_df['normalized_cost'] = normalized_cost
sim_df['relative_difference'] = calculate_relative_distance(sim_df, experiments_df)

# ok = [check_cost(i) for i in range(len(sim_df))]
# diff = sim_df['best_energy_by_sample']*(1/experiments_df['b']) - experiments_df['exact_distance']
# sim_df['ok'] = ok
# sim_df['diff'] = diff
# sim_df['new_diff'] = np.where(sim_df['ok'] < 1000, sim_df['diff'], experiments_df['vertices']**2)
# sim_df['v'] = experiments_df['vertices']
# sim_df['diff'] = diff.abs()
# sim_df['v'] = experiments_df['vertices']
#
# sim_df.to_pickle(SIM_PATH)
#
b_df = pd.DataFrame()
for a, b in bs:
    b_df[f'b{b}'] = sim_df[experiments_df['b'] < b + 0.0001][experiments_df['b'] > b - 0.0001].groupby(['v'])['relative_difference'].mean()

b_df = b_df.round(decimals=2)
print(b_df)


#
# ls = []
# for i in range(len(sim_df)):
#     sim = sim_df.loc[i]
#     l = len(sim['best_sample'].keys())
#     ls.append(l)
# sim_df['l'] = ls
