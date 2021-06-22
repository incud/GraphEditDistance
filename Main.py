from SmallGraphDataset import SmallGraphDataset
from GraphEditDistanceCalculator import GraphEditDistanceCalculator
from os import path
import pandas as pd
from datetime import datetime

PATH = "graph_experiments/dwave_experiments.pickle"
if path.exists(PATH):
    pd = pd.read_pickle(PATH)
    print("Loaded from file")
else:
    COLUMNS = ["num_source_variables", "num_target_variables", "max_chain_length", "sum_chain_length", "chain_strength", "chain_break_method",
               "solver", "machine", "best_sample", "best_energy", "mean_energy", "best_sample_pp", "best_energy_pp", "mean_energy_pp",
               'vertices', 'g1_name', 'g2_name', 'a', 'b', 'start', 'end']
    pd = pd.DataFrame(columns=COLUMNS)
    print("Created new dataframe")


dataset = SmallGraphDataset()

for vertices in dataset.get_graphs_vertices_count():
    for g1_index in range(dataset.get_graphs_count(vertices)):
        for g2_index in range(dataset.get_graphs_count(vertices)):
            g1 = dataset.get_graph_object(vertices, g1_index)
            g2 = dataset.get_graph_object(vertices, g2_index)
            n1 = dataset.get_graph_name(vertices, g1_index)
            n2 = dataset.get_graph_name(vertices, g2_index)
            print("NEW INSTANCE", n1, n2, flush=True)
            a_b_values = [(1, 0.1), (1, 0.05), (1, 0.01), (1, 0.005)]
            for (a, b) in a_b_values:
                print(f"a={a}, b={b}: ", end="", flush=True)

                calculator = GraphEditDistanceCalculator(g1=g1, g2=g2, a=a, b=b)

                start = datetime.now()
                sample, energy, info = calculator.run_simulated()
                end = datetime.now()

                row = {'vertices': vertices, 'g1_name': n1, 'g2_name': n2, 'start': start, 'end': end, 'a': a, 'b': b}
                row = {**row, **info}
                pd.loc[len(pd)] = row
                print(".", end="", flush=True)
                pd.to_pickle(PATH)

                for is_adv, is_leap in [(False, False), (True, False), (False, True)]:
                    calculator = GraphEditDistanceCalculator(g1=g1, g2=g2, a=a, b=b)

                    start = datetime.now()
                    sample, energy, info = calculator.run_dwave(is_advantage=is_adv, is_leap=is_leap)
                    end = datetime.now()

                    row = {'vertices': vertices, 'g1_name': n1, 'g2_name': n2, 'start': start, 'end': end}
                    row = {**row, **info}
                    pd.loc[len(pd)] = row
                    print(".", end="", flush=True)
                    pd.to_pickle(PATH)

                print("END", flush=True)
                pd.to_pickle(PATH)

pd.to_pickle(PATH)
















# g1 = dataset.get_graph_object(4, 0)
# g2 = dataset.get_graph_object(4, 1)
# n1 = dataset.get_graph_name(4, 0)
# n2 = dataset.get_graph_name(4, 1)
# print(n1, "vs", n2)
# distance_process = GraphEditDistanceSolver(g1, g2, a=1.0, b=0.1)
# result = distance_process.run_dwave(is_advantage=True)
# import dwave.inspector
# sample, energy, more_info = result
# sample_set = more_info['sample_set']
# bqm = solver.solver.bqm
# sampler = solver.solver.sampler
# dwave.inspector.show(sample_set, bqm, sampler)
