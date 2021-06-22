from SmallGraphDataset import SmallGraphDataset
from GraphEditDistanceCalculator import GraphEditDistanceCalculator
from os import path
import pandas
from datetime import datetime

PATH = "graph_experiments/dwave_experiments.pickle"
if path.exists(PATH):
    df = pandas.read_pickle(PATH)
    print("Loaded from file")
else:
    COLUMNS = ["num_source_variables", "num_target_variables", "max_chain_length", "sum_chain_length", "chain_strength", "chain_break_method",
               "solver", "machine", "best_sample", "best_energy", "mean_energy", "best_sample_pp", "best_energy_pp", "mean_energy_pp",
               'vertices', 'g1_name', 'g2_name', 'a', 'b', 'start', 'end']
    df = pandas.DataFrame(columns=COLUMNS)
    print("Created new dataframe")

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

dataset = SmallGraphDataset()

for vertices in dataset.get_graphs_vertices_count():

    if vertices == 3:
        print("Vertici 3 - Gia fatti")
        continue

    for g1_index in range(dataset.get_graphs_count(vertices)):

        for g2_index in range(dataset.get_graphs_count(vertices)):

            if vertices == 4 and g1_index < 6:
                n1 = dataset.get_graph_name(vertices, g1_index)
                print("Vertici 4 - Gia fatto indice: ", g1_index, n1)
                continue

            if vertices == 4 and g1_index == 6 and g2_index < 3:
                n1 = dataset.get_graph_name(vertices, g1_index)
                n2 = dataset.get_graph_name(vertices, g2_index)
                print("Vertici 4 - Gia fatto indice: ", g1_index, n1, "vs", g2_index, n2)
                continue

            g1 = dataset.get_graph_object(vertices, g1_index)
            g2 = dataset.get_graph_object(vertices, g2_index)
            n1 = dataset.get_graph_name(vertices, g1_index)
            n2 = dataset.get_graph_name(vertices, g2_index)
            print("NEW INSTANCE", vertices, n1, n2, flush=True)
            a_b_values = [(1, 0.1), (1, 0.05), (1, 0.01), (1, 0.005)]
            for (a, b) in a_b_values:
                print(f"a={a}, b={b}: ", end="", flush=True)

                calculator = GraphEditDistanceCalculator(g1=g1, g2=g2, a=a, b=b)

                start = datetime.now()
                sample, energy, info = calculator.run_simulated()
                end = datetime.now()

                row = {'vertices': vertices, 'g1_name': n1, 'g2_name': n2, 'start': start, 'end': end, 'a': a, 'b': b}
                row = {**row, **info}
                df.loc[len(df)] = row
                print(".", end="", flush=True)
                df.to_pickle(PATH)

                for is_adv, is_leap in [(False, False), (True, False), (False, True)]:
                    calculator = GraphEditDistanceCalculator(g1=g1, g2=g2, a=a, b=b)

                    start = datetime.now()
                    sample, energy, info = calculator.run_dwave(is_advantage=is_adv, is_leap=is_leap)
                    end = datetime.now()

                    row = {'vertices': vertices, 'g1_name': n1, 'g2_name': n2, 'start': start, 'end': end, 'a': a, 'b': b}
                    row = {**row, **info}
                    df.loc[len(df)] = row
                    print(".", end="", flush=True)
                    df.to_pickle(PATH)

                print("END", flush=True)
                df.to_pickle(PATH)

df.to_pickle(PATH)
















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
