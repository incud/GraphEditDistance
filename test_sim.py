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

sim = pd.read_pickle("data/simulated_annealing_solutions.pickle")['best_energy_by_sample']

g = pd.read_pickle("data/experiments.pickle")
ged = g['exact_distance']
for i in range (len(g)):
    sim.iloc[i] = sim.iloc[i]/g.iloc[i]['b']


