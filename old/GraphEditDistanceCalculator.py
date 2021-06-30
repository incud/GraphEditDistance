from PyQuboOptimizer import PyQuboOptimizer
from QaoaOptimizer import QaoaOptimizer
from DwaveOptimizer import DwaveOptimizer
from pyqubo import Array
from functools import reduce
import networkx as nx


class GraphEditDistanceCalculator:

    def __init__(self, g1, g2, a=1, b=0.1):
        assert len(g1.nodes) == len(g2.nodes), "I grafi devono avere lo stesso numero di vertici"
        self.g1 = g1
        self.g2 = g2
        self.A = a
        self.B = b
        self.solver = None
        self.symbols = self.define_symbols()
        self.H = self.A * self.hard_constraint(self.symbols) + self.B * self.soft_constraint(self.symbols)

    def define_symbols(self):
        n = len(self.g1.nodes)
        return Array.create('x', shape=(n, n), vartype='BINARY')

    def hard_constraint(self, symbols):
        H = 0
        for node1 in self.g1.nodes:
            H_term = reduce(lambda H, t: H + t, symbols[node1, :], -1)  # (-1 + x_a_0 + x_a_1 + ... + x_a_n)
            H = H + (H_term) ** 2
        for node2 in self.g2.nodes:
            H_term = reduce(lambda H, t: H + t, symbols[:, node2], -1)  # (-1 + x_0_b + x_1_b + ... + x_n_b)
            H = H + (H_term) ** 2
        return H

    def soft_constraint(self, symbols):
        H1 = sum([symbols[a, b] * symbols[c, d] for (a, b) in nx.complement(self.g1).edges for (c, d) in self.g2.edges])
        H2 = sum([symbols[a, b] * symbols[c, d] for (a, b) in self.g1.edges for (c, d) in nx.complement(self.g2).edges])
        return H1 + H2

    def run_simulated(self):
        self.solver = PyQuboOptimizer(self.H)
        return self.run()

    def run_qaoa(self, p=1):
        self.solver = QaoaOptimizer(self.H, p=p)
        return self.run()

    def run_dwave(self, is_advantage=False, is_leap=False):
        self.solver = DwaveOptimizer(self.H, is_advantage=is_advantage, is_leap=is_leap)
        return self.run()

    def run(self):
        the_sample, the_energy = self.solver.solve()
        more_info = self.solver.get_record_result()
        return the_sample, the_energy, more_info

    def get_record_result(self):
        return self.solver.get_record_result()
