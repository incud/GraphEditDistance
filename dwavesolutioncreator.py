import warnings
warnings.filterwarnings("ignore")

from os import path
import pandas as pd
from experimentcreator import generate_hamiltonian

from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler
from greedy import SteepestDescentSolver


def generate_dwave_dataframe(the_path):
    if path.exists(the_path):
        print("generate_dwave_dataframe: Loaded from file")
        return pd.read_pickle(the_path)
    else:
        columns = ["experiment", "start", "end",
                   "best_sample", "best_energy", "best_energy_by_sample",
                   "best_sample_pp", "best_energy_pp", "best_energy_by_sample_pp",
                   "num_source_variables", "num_target_variables", "max_chain_length", "chain_strength", "chain_break_method"]
        return pd.DataFrame(columns=columns)


def run_dwave_2000_experiment(the_experiment, queue):
    run_dwave_experiment(the_experiment, 'DW_2000Q_6', queue)


def run_dwave_advantage_experiment(the_experiment, queue):
    run_dwave_experiment(the_experiment, 'Advantage_system1.1', queue)


def run_dwave_experiment(experiment, machine, queue):

    bqm = experiment["bqm"]
    model = generate_hamiltonian(experiment["g1"], experiment["g2"], experiment["a"], experiment["b"]).compile()

    sampler = EmbeddingComposite(DWaveSampler(solver=machine))
    sample_set = sampler.sample(bqm, num_reads=1000, answer_mode='raw', return_embedding=True)
    decoded_samples = model.decode_sampleset(sample_set)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    # =========================================================================
    # get embedding info (method inspired by: from dwave.inspector.adapters import _problem_stats)
    embedding_context = sample_set.info['embedding_context']
    embedding = embedding_context.get('embedding')
    chain_strength = embedding_context.get('chain_strength')
    chain_break_method = embedding_context.get('chain_break_method')
    num_source_variables = len(sample_set.variables)
    # best guess for target variables
    if sample_set and embedding:
        target_vars = {t for s in sample_set.variables for t in embedding[s]}
        num_target_variables = len(target_vars)
    else:
        target_vars = set()
        num_target_variables = None

    # max chain length
    if embedding:
        # consider only active variables in response
        # (so fixed embedding won't falsely increase the max chain len)
        max_chain_length = max(len(target_vars.intersection(chain))
                               for chain in embedding.values())
    else:
        max_chain_length = 1

    # =========================================================================
    # Post-processing: see https://docs.ocean.dwavesys.com/en/stable/examples/pp_greedy.html#pp-greedy
    solver_greedy = SteepestDescentSolver()
    sample_set_pp = solver_greedy.sample(bqm, initial_states=sample_set)
    decoded_samples_pp = model.decode_sampleset(sample_set_pp)
    best_sample_pp = min(decoded_samples_pp, key=lambda x: x.energy)

    row = {
            "best_sample": best_sample.sample,
            "best_energy": best_sample.energy,
            "best_energy_by_sample": bqm.energy(best_sample.sample),
            "best_sample_pp": best_sample_pp.sample,
            "best_energy_pp": best_sample_pp.energy,
            "best_energy_by_sample_pp": bqm.energy(best_sample_pp.sample),
            "num_source_variables": num_source_variables,
            "num_target_variables": num_target_variables,
            "max_chain_length": max_chain_length,
            "chain_strength": chain_strength,
            "chain_break_method": chain_break_method
    }
    queue.put(row)


def run_dwave_leap_experiment(experiment, queue):

    bqm = experiment["bqm"]
    model = generate_hamiltonian(experiment["g1"], experiment["g2"], experiment["a"], experiment["b"]).compile()

    sampler = LeapHybridSampler()
    sample_set = sampler.sample(bqm)
    decoded_samples = model.decode_sampleset(sample_set)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    # Post-processing: see https://docs.ocean.dwavesys.com/en/stable/examples/pp_greedy.html#pp-greedy
    solver_greedy = SteepestDescentSolver()
    sample_set_pp = solver_greedy.sample(bqm, initial_states=sample_set)
    decoded_samples_pp = model.decode_sampleset(sample_set_pp)
    best_sample_pp = min(decoded_samples_pp, key=lambda x: x.energy)

    row = {
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
    queue.put(row)
