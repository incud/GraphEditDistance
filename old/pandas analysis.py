df = pandas.concat( [df1, df2]) # no axis=1
df.reset_index(drop=True, inplace=True)

from SmallGraphDataset import SmallGraphDataset
smd = SmallGraphDataset()
df['g1'] = df.apply(lambda x: smd.get_graph_object_by_name(x['vertices'], x['g1_name']), axis=1)
df['g2'] = df.apply(lambda x: smd.get_graph_object_by_name(x['vertices'], x['g2_name']), axis=1)

from GraphEditDistanceCalculator import GraphEditDistanceCalculator
from PyQuboOptimizer import PyQuboOptimizer
def calculate_real_energy(x):
    gedc = GraphEditDistanceCalculator(x['g1'], x['g2'], x['a'], x['b'])
    gedc.solver = PyQuboOptimizer(gedc.H)
    sample = x['best_sample_pp']
    if not isinstance(x['best_sample_pp'], dict):
        sample = x['best_sample']
    energy = gedc.solver.evaluate(sample)
    if 0.0000001 > energy > -0.0000001:
        energy = 0
    return energy