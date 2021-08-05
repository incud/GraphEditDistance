from os import path
from pyqubo import Array
from networkx.algorithms.similarity import graph_edit_distance
from functools import reduce
import pandas as pd
import networkx as nx


def generate_hamiltonian(g1, g2, a, b):
    n = len(g1.nodes)
    symbols = Array.create('x', shape=(n, n), vartype='BINARY')

    ## hard constraint
    #H = 0
    #for node1 in g1.nodes:
    #    H_term = reduce(lambda h, t: h + t, symbols[node1, :], -1)  # (-1 + x_a_0 + x_a_1 + ... + x_a_n)
    #    H = H + (H_term) ** 2
    #for node2 in g2.nodes:
    #    H_term = reduce(lambda h, t: h + t, symbols[:, node2], -1)  # (-1 + x_0_b + x_1_b + ... + x_n_b)
    #    H = H + (H_term) ** 2
    ## soft constraint
    #H1 = sum([symbols[u, i] * symbols[v, j] for (i, j) in nx.complement(g1).edges for (u, v) in g2.edges])
    #H2 = sum([symbols[u, i] * symbols[v, j] for (i, j) in g1.edges for (u, v) in nx.complement(g2).edges])
    ## compose the Hs

    ##Formulation_v2 (QUBOGraphIso.pdf):
    ##hard constraint:
    #H = 0
    #for node1 in g1.nodes:
    #    H_term = reduce(lambda h, t: h + t, symbols[node1, :], -1)  # (-1 + x_a_0 + x_a_1 + ... + x_a_n)
    #    H = H + (H_term) ** 2
    #for node2 in g2.nodes:
    #    H_term = reduce(lambda h, t: h + t, symbols[:, node2], -1)  # (-1 + x_0_b + x_1_b + ... + x_n_b)
    #    H = H + (H_term) ** 2
    #
    ##Soft constraint:
    #P = 0
    #for (i_g1, j_g1) in g1.edges:
    #    temp_i = 0
    #    for i_g2 in g2.nodes:
    #        temp_j = 0
    #        for j_g2 in g2.nodes:
    #            e_i_j = int(g2.has_edge(i_g2, j_g2))
    #            temp_j += temp_j + (symbols[j_g1, j_g2] * (1 - e_i_j))
    #        temp_i = temp_i + (symbols[i_g1, i_g2] * temp_j)
    #    P = P + temp_i
    #
    #for (i_g1, j_g1) in g2.edges:
    #    temp_i = 0
    #    for i_g2 in g1.nodes:
    #        temp_j = 0
    #        for j_g2 in g1.nodes:
    #            e_i_j = int(g2.has_edge(i_g2, j_g2))
    #            temp_j += temp_j + (symbols[j_g1, j_g2] * (1 - e_i_j))
    #        temp_i = temp_i + (symbols[i_g1, i_g2] * temp_j)
    #    P = P + temp_i
    #
    #return a * H + b * P

def generate_hamiltonian(g1, g2, a, b):
    n = len(g1.nodes)
    symbols = Array.create('x', shape=(n, n), vartype='BINARY')

    # hard constraint
    H = 0
    for node1 in g1.nodes:
        H_term = reduce(lambda h, t: h + t, symbols[node1, :], -1)  # (-1 + x_a_0 + x_a_1 + ... + x_a_n)
        H = H + (H_term) ** 2
    for node2 in g2.nodes:
        H_term = reduce(lambda h, t: h + t, symbols[:, node2], -1)  # (-1 + x_0_b + x_1_b + ... + x_n_b)
        H = H + (H_term) ** 2
    # soft constraint
    H1 = 0
    for e in g1.edges:
        H1 += generate_edge_predicate(symbols, e, g2)
    H2 = 0
    for e in g2.edges:
        H2 += generate_edge_predicate(symbols, e, g1)

    # compose the Hs
    return a * H + b * (H1 + H2)

def generate_edge_predicate(x, edge, g):
    #print(g.edges)
    term = 0
    i, j = edge
    for ip in g.nodes:
        for jp in g.nodes:
            #print(f"i:{i}, j:{j}, ip: {ip}, jp:{jp}")
            e_ip_jp = g.has_edge(ip, jp)
            #print(f"{g.edges} has ({ip,jp}) : {e_ip_jp}")
            if not e_ip_jp:
                term += x[i][ip] * x[j][jp]
                #print(f"add term {x[i][ip]*x[j][jp]}")
            # term += x[i][ip] * x[j][jp] * (1 - e_ip_jp)
    return term


def generate_experiments_dataframe(the_path, the_graph_df):
    if path.exists(the_path):
        print("Loaded from file")
        return pd.read_pickle(the_path)
    else:
        columns = ["vertices", "g1_index", "g2_index", "g1", "g2", "a", "b", "bqm", "exact_distance"]
        the_experiments_df = pd.DataFrame(columns=columns)

        vertices = the_graph_df['vertices'].unique()

        for v in [3, 4, 5]:
            print(f"=================== VERTICES {v} ===================")
            has_v_vertices = the_graph_df['vertices'] == v
            v_graph_df = the_graph_df[has_v_vertices]
            for i in range(len(v_graph_df)):
                for j in range(len(v_graph_df)):
                    print(f"i={i}, j={j}: ", end="")
                    g1, g2 = v_graph_df.iloc[i]["g"], v_graph_df.iloc[j]["g"]
                    exact_distance = graph_edit_distance(g1, g2)
                    print(f"distance={exact_distance} ", end="")
                    for a, b in [(1, 0.1), (1, 0.05), (1, 0.01)]:
                        h = generate_hamiltonian(g1, g2, a, b)
                        bqm = h.compile().to_bqm()
                        row = {"vertices": v, "g1_index": i, "g2_index": j, "g1": g1, "g2": g2,
                               "a": a, "b": b, "exact_distance": exact_distance, "bqm": bqm}
                        the_experiments_df.loc[len(the_experiments_df)] = row
                    print()

        return the_experiments_df
