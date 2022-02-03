import warnings
warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
import multiprocessing
from datetime import datetime
import pandas as pd

qaoa_dfs = []
for p in range(5):
    qaoa_dfs.append(pd.read_pickle(f"circuit_stats/qaoa_{p}_stats.pickle"))

for p in range(5):
    qaoa_dfs[p]['parameters'] = [(p+1)*2] * len(qaoa_dfs[p])

for p in range(5):
    qaoa_dfs[p] = qaoa_dfs[p].drop('time', axis=1)

qaoa_recap_dfs = []
for p in range(5):
    qaoa_recap_dfs.append(qaoa_dfs[p].drop('experiment', axis=1).fillna(0).groupby(['v']).mean())

