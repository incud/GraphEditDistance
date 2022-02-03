import warnings
warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
import multiprocessing
from datetime import datetime
import pandas


from generate_vqe_resources_helper import run_qaoa_stats, run_vqe_stats


pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
# genera i grafi
GRAPHS_PATH = "data/graphs.pickle"
graph_df = generate_graph_dataframe(GRAPHS_PATH)
# genera gli esperimenti
EXPERIMENTS_PATH = "data/experiments.pickle"
experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df)

vqe_p1_df = pandas.DataFrame(columns=["experiment", "v", "qubits", "parameters", "depth", "size", "time"])
vqe_p2_df = pandas.DataFrame(columns=["experiment", "v", "qubits", "parameters", "depth", "size", "time"])
vqe_p3_df = pandas.DataFrame(columns=["experiment", "v", "qubits", "parameters", "depth", "size", "time"])
vqe_p4_df = pandas.DataFrame(columns=["experiment", "v", "qubits", "parameters", "depth", "size", "time"])
vqe_p5_df = pandas.DataFrame(columns=["experiment", "v", "qubits", "parameters", "depth", "size", "time"])
vqe_dfs = [vqe_p1_df, vqe_p2_df, vqe_p3_df, vqe_p4_df, vqe_p5_df]


queue = multiprocessing.Queue()

for p in [5]:
    max_iter = 1000
    for i in range(len(experiments_df)):

        if i % 3 > 0:
            continue

        if (i//3) % 5 > 0:
            continue

        experiment = experiments_df.loc[i]

        # fai partire il simulatore in un nuovo processo (pulizia memoria utilizzata ottimale)
        process = multiprocessing.Process(target=run_vqe_stats, args=(experiment, queue, i, p))
        start = datetime.now()
        process.start()
        process.join()
        end = datetime.now()

        row = queue.get()
        row['time'] = end - start
        print(row)

        vqe_dfs[p].loc[len(vqe_dfs[p])] = row
        vqe_dfs[p].to_pickle(f"circuit_stats/vqe_{p}_stats.pickle")
