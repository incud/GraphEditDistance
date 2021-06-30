from os import path
from random import choice
import pandas as pd
import networkx as nx


def generate_graph(vertices, p=0.1):
    g = nx.fast_gnp_random_graph(vertices, p, directed=False)
    # assure graph is connected
    for node in g.nodes:
        neighbors = list(g.neighbors(node))
        if not neighbors:  # se non ho vicini collego il nodo con un altro a caso
            other_node = choice(list(set(g.nodes).difference([node])))
            g.add_edge(node, other_node)
    # remove auto-loops
    for node in g.nodes:
        if g.has_edge(node, node):
            g.remove_edge(node, node)
    return g


def generate_graph_dataframe(the_path):
    if path.exists(the_path):
        print("Loaded from file")
        return pd.read_pickle(the_path)
    else:
        columns = ["vertices", "name", "p", "g"]
        the_df = pd.DataFrame(columns=columns)
        for vertices in [3, 4, 5, 6, 7, 8, 9]:
            for p in [0.1, 0.33, 0.66, 0.9]:
                g = generate_graph(vertices, p)
                row = {"vertices": vertices, "name": f"graph_{vertices}_{p}", "p": p, "g": g}
                the_df.loc[len(the_df)] = row
        return the_df
