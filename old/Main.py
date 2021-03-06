from SmallGraphDataset import SmallGraphDataset
from os import path
import pandas
import multiprocessing
from worker_process import worker

PATH = "graph_experiments/dwave_all.pickle"
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)


def load_experiments():
    if path.exists(PATH):
        df = pandas.read_pickle(PATH)
        print("Loaded from file")
    else:
        COLUMNS = ["num_source_variables", "num_target_variables", "max_chain_length", "sum_chain_length",
                   "chain_strength", "chain_break_method",
                   "solver", "machine", "best_sample", "best_energy", "mean_energy", "best_sample_pp", "best_energy_pp",
                   "mean_energy_pp",
                   'vertices', 'g1_name', 'g2_name', 'a', 'b', 'start', 'end']
        df = pandas.DataFrame(columns=COLUMNS)
        print("Created new dataframe")
    return df


def manager_process():
    df = load_experiments()
    dataset = SmallGraphDataset()
    queue = multiprocessing.Queue()

    for vertices in dataset.get_graphs_vertices_count():

        if vertices == 3 or vertices == 4:
            print("Vertici", vertices, "- Gia fatti")
            continue

        for g1_index in range(dataset.get_graphs_count(vertices)):

            for g2_index in range(dataset.get_graphs_count(vertices)):

                if vertices == 5 and g1_index < 5:
                    n1 = dataset.get_graph_name(vertices, g1_index)
                    print("Vertici 5 - Gia fatto indice: ", g1_index, n1)
                    continue

                if vertices == 5 and g1_index == 5 and g2_index < 9:
                    n1 = dataset.get_graph_name(vertices, g1_index)
                    n2 = dataset.get_graph_name(vertices, g2_index)
                    print("Vertici 5 - Gia fatto indice: ", g1_index, n1, "vs", g2_index, n2)
                    continue

                g1 = dataset.get_graph_object(vertices, g1_index)
                g2 = dataset.get_graph_object(vertices, g2_index)
                n1 = dataset.get_graph_name(vertices, g1_index)
                n2 = dataset.get_graph_name(vertices, g2_index)
                print("NEW INSTANCE", vertices, n1, n2, flush=True)
                a_b_values = [(1, 0.1), (1, 0.05), (1, 0.01), (1, 0.005)]
                for (a, b) in a_b_values:
                    print(f"a={a}, b={b}: ", end="", flush=True)

                    p = multiprocessing.Process(target=worker, args=(vertices, g1, g2, n1, n2, a, b, queue))
                    p.start()
                    p.join()

                    rows = queue.get()

                    for row in rows:
                        df.loc[len(df)] = row

                    df.to_pickle(PATH)












import networkx as nx
from pyqubo import Array

g1 = nx.from_edgelist([(0, 1), (0, 2)])
g2 = nx.from_edgelist([(0, 1), (0, 2), (1, 2)])
symbols = Array.create('x', shape=(3, 3), vartype='BINARY')
not_in_g1 = nx.complement(g1).edges
in_g2 = g2.edges
print(f"Not in G1={not_in_g1}")
print(f"In G2={in_g2}")

H = 0
for (i, j) in not_in_g1:
    for (u, v) in in_g2:
        # per confrontare E1=(i,j) con E2=(u,v)
        # devo fare x[i][u]*x[j][v] ed anche lo stesso termine con u,v invertiti
        term = (symbols[i, u] * symbols[j, v] + symbols[i, v] * symbols[j, u])
        print(f"Adding term {term}")
        H += term
print(H)

model = H.compile()
bqm = model.to_bqm()

import neal
sampler = neal.SimulatedAnnealingSampler()
sample_set = sampler.sample(bqm, num_reads=10000, answer_mode='raw', return_embedding=True)
decoded_samples = model.decode_sampleset(sample_set)
best_sample = min(decoded_samples, key=lambda x: x.energy)

the_sample_for_calculating_energy = best_sample.sample
bqm.energy(the_sample_for_calculating_energy)









