from dimod.reference.samplers import ExactSolver
from neal import SimulatedAnnealingSampler
import numpy as np
np.set_printoptions(precision=3, suppress=True)


class PyQuboOptimizer:

    def __init__(self, hamiltonian, is_simulated_annealing=True):
        self.H = hamiltonian
        self.is_simulated_annealing = is_simulated_annealing
        self.best_sample = None
        self.best_energy = None

    def solve(self):
        model = self.H.compile()
        bqm = model.to_bqm()

        if self.is_simulated_annealing:
            sample_set = SimulatedAnnealingSampler().sample(bqm, num_reads=1000)
        else:
            sample_set = ExactSolver().sample(bqm)

        decoded_samples = model.decode_sampleset(sample_set)
        best_sample = min(decoded_samples, key=lambda x: x.energy)

        self.best_sample = best_sample.sample
        self.best_energy = best_sample.energy
        return self.best_sample, self.best_energy

    def evaluate(self, sample):
        instance = {var: -1 if val == 0 or val == -1 else 1 for var, val in sample.items()}
        # print(instance)
        linear, quadratic, offset = self.H.compile().to_ising()
        cost = offset
        for term, coeff in linear.items():
            # print(term, coeff, instance[term] * coeff)
            cost += instance[term] * coeff
        for (term1, term2), coeff in quadratic.items():
            # print(term1, term2, coeff, instance[term1] * instance[term2] * coeff)
            cost += instance[term1] * instance[term2] * coeff
        return cost

    def get_more_info(self):
        return None

    def get_record_result(self):
        return {
            "solver": "simulated_annealing" if self.is_simulated_annealing else "exact",
            "best_sample": self.best_sample,
            "best_energy": self.best_energy
        }
