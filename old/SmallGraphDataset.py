import networkx as nx


class SmallGraphDataset:

    def __init__(self):
        graphs = {}

        # 3 vertici
        N = 3
        graphs[N] = []
        graphs[N].append((nx.empty_graph(N), "empty"))
        graphs[N].append((self.create_graph([(0, 1)], [2]), "segment"))
        graphs[N].append((nx.complete_graph(N), "complete"))

        # 4 vertici
        N = 4
        graphs[N] = []
        graphs[N].append((nx.empty_graph(N), "empty"))
        graphs[N].append((self.create_graph([(0, 1)], [2, 3]), "segment"))
        graphs[N].append((self.create_graph([(0, 1), (1, 2)], [3]), "angle"))
        graphs[N].append((nx.Graph([(0, 1), (1, 2), (2, 3)]), "U"))
        graphs[N].append((nx.cycle_graph(N), "cycle"))
        graphs[N].append((self.create_graph([(0, 1), (1, 2), (2, 0)], [3]), "triangle"))
        graphs[N].append((nx.star_graph(N-1), "star"))
        graphs[N].append((nx.complete_graph(N), "complete"))

        # 5 vertici
        N = 5
        graphs[N] = []
        graphs[N].append((nx.empty_graph(N), "empty"))
        graphs[N].append((nx.cycle_graph(N), "cycle"))
        graphs[N].append((nx.star_graph(N-1), "star"))
        graphs[N].append((nx.complete_graph(N), "complete"))
        graphs[N].append((nx.gnm_random_graph(N, 1, seed=12), "random1-1"))
        graphs[N].append((nx.gnm_random_graph(N, 2, seed=34), "random2-1"))
        graphs[N].append((nx.gnm_random_graph(N, 2, seed=56), "random2-2"))
        graphs[N].append((nx.gnm_random_graph(N, 3, seed=78), "random3-1"))
        graphs[N].append((nx.gnm_random_graph(N, 3, seed=11), "random3-2"))
        graphs[N].append((nx.gnm_random_graph(N, 3, seed=22), "random3-3"))
        graphs[N].append((nx.gnm_random_graph(N, 4, seed=33), "random4-1"))
        graphs[N].append((nx.gnm_random_graph(N, 4, seed=44), "random4-2"))
        graphs[N].append((nx.gnm_random_graph(N, 4, seed=55), "random4-3"))
        graphs[N].append((nx.gnm_random_graph(N, 4, seed=66), "random4-4"))
        graphs[N].append((nx.gnm_random_graph(N, 5, seed=77), "random5-1"))
        graphs[N].append((nx.gnm_random_graph(N, 5, seed=88), "random5-2"))
        graphs[N].append((nx.gnm_random_graph(N, 5, seed=99), "random5-3"))
        graphs[N].append((nx.gnm_random_graph(N, 5, seed=13), "random5-4"))
        graphs[N].append((nx.gnm_random_graph(N, 5, seed=14), "random5-5"))
        graphs[N].append((nx.gnm_random_graph(N, 8, seed=77), "random8-1"))
        graphs[N].append((nx.gnm_random_graph(N, 8, seed=88), "random8-2"))
        graphs[N].append((nx.gnm_random_graph(N, 8, seed=99), "random8-3"))
        graphs[N].append((nx.gnm_random_graph(N, 8, seed=13), "random8-4"))
        graphs[N].append((nx.gnm_random_graph(N, 8, seed=14), "random8-5"))

        N = 6
        graphs[N] = []
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.2, seed=5434254), "graph_v6_r1"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.4, seed=1264363), "graph_v6_r2"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.6, seed=4280230), "graph_v6_r3"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.8, seed=1456479), "graph_v6_r4"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.9, seed=6895675), "graph_v6_r5"))

        N = 7
        graphs[N] = []
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.2, seed=5434254), "graph_v7_r1"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.4, seed=1264363), "graph_v7_r2"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.6, seed=4280230), "graph_v7_r3"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.8, seed=1456479), "graph_v7_r4"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.9, seed=6895675), "graph_v7_r5"))

        N = 8
        graphs[N] = []
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.2, seed=5434254), "graph_v8_r1"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.4, seed=1264363), "graph_v8_r2"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.6, seed=4280230), "graph_v8_r3"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.8, seed=1456479), "graph_v8_r4"))
        graphs[N].append((nx.gnm_random_graph(N, (N // 2) * 0.9, seed=6895675), "graph_v8_r5"))

        N = 9
        graphs[N] = []
        graphs[N].append((nx.gnm_random_graph(N, (N//2)*0.2, seed=5647587), "graph_v9_r1"))
        graphs[N].append((nx.gnm_random_graph(N, (N//2)*0.4, seed=2355904), "graph_v9_r2"))
        graphs[N].append((nx.gnm_random_graph(N, (N//2)*0.6, seed=6557575), "graph_v9_r3"))
        graphs[N].append((nx.gnm_random_graph(N, (N//2)*0.8, seed=1934039), "graph_v9_r4"))
        graphs[N].append((nx.gnm_random_graph(N, (N//2)*0.9, seed=1029349), "graph_v9_r5"))

        self.graphs = graphs
        self.check_correctness()

    @staticmethod
    def create_graph(edge_list, other_nodes):
        g = nx.Graph(edge_list)
        for node in other_nodes:
            g.add_node(node)
        return g

    def get_graphs_vertices_count(self):
        return list(self.graphs.keys())

    def get_graphs_count(self, v):
        return len(self.graphs[v])

    def get_graph_object(self, v, i):
        return self.graphs[v][i][0]

    def get_graph_object_by_name(self, v, name):
        grs = self.graphs[v]
        for graph_obj, graph_name in grs:
            if graph_name == name:
                return graph_obj

    def get_graph_name(self, v, i):
        return self.graphs[v][i][1]

    def check_correctness(self):
        for vertex in self.get_graphs_vertices_count():
            for g1_index in range(self.get_graphs_count(vertex)):
                for g2_index in range(self.get_graphs_count(vertex)):
                    g1 = self.get_graph_object(vertex, g1_index)
                    g2 = self.get_graph_object(vertex, g2_index)
                    n1 = self.get_graph_name(vertex, g1_index)
                    n2 = self.get_graph_name(vertex, g2_index)
                    if len(g1.nodes) != len(g2.nodes):
                        raise ValueError(f"{vertex}: {len(g1.nodes)} . {n1} != {len(g2.nodes)} {n2}")
