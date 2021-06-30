from PyQuboOptimizer import PyQuboOptimizer
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler
from greedy import SteepestDescentSolver

class DwaveOptimizer(PyQuboOptimizer):

    def __init__(self, hamiltonian, is_advantage=False, is_leap=False):
        super(DwaveOptimizer, self).__init__(hamiltonian)
        if is_advantage and is_leap:
            raise ValueError("Cannot be both is_advantage and is_leap")
        self.is_advantage = is_advantage
        self.is_leap = is_leap
        self.bqm = None
        self.sampler = None
        self.sample_set = None
        self.sample_set_pp = None
        self.embedding_context = None
        self.model = None
        if self.is_advantage:
            self.machine = 'Advantage_system1.1'
        elif self.is_leap:
            self.machine = 'Leap_Hybrid'
        else:
            self.machine = 'DW_2000Q_6'

    def solve(self):
        self.model = self.H.compile()
        self.bqm = self.model.to_bqm()

        if self.is_leap:
            self.solve_leap()
        else:
            self.solve_dwave()

        # Post-processing: see https://docs.ocean.dwavesys.com/en/stable/examples/pp_greedy.html#pp-greedy
        solver_greedy = SteepestDescentSolver()
        self.sample_set_pp = solver_greedy.sample(self.bqm, initial_states=self.sample_set)
        decoded_samples = self.model.decode_sampleset(self.sample_set_pp)
        best_sample = min(decoded_samples, key=lambda x: x.energy)
        return best_sample.sample, best_sample.energy

    def solve_leap(self):
        self.sampler = LeapHybridSampler()
        self.sample_set = self.sampler.sample(self.bqm)
        self.embedding_context = None

    def solve_dwave(self):
        self.sampler = EmbeddingComposite(DWaveSampler(solver=self.machine))
        self.sample_set = self.sampler.sample(self.bqm, num_reads=1000, answer_mode='raw', return_embedding=True)
        # get embedding info
        self.embedding_context = self.sample_set.info['embedding_context']

    def get_stats(self):
        if not self.is_leap:
            # method inspired by: from dwave.inspector.adapters import _problem_stats
            embedding = self.embedding_context.get('embedding')
            chain_strength = self.embedding_context.get('chain_strength')
            chain_break_method = self.embedding_context.get('chain_break_method')

            num_source_variables = len(self.sample_set.variables)

            # best guess for target variables
            if self.sample_set and embedding:
                target_vars = {t for s in self.sample_set.variables for t in embedding[s]}
                num_target_variables = len(target_vars)
            else:
                target_vars = set()
                num_target_variables = None

            # max chain length
            if embedding:
                # consider only active variables in response
                # (so fixed embedding won't falsely increase the max chain len)
                max_chain_length = max(len(target_vars.intersection(chain))
                                       for chain in embedding.values())
                sum_chain_length = sum(len(target_vars.intersection(chain))
                                       for chain in embedding.values())
            else:
                max_chain_length = 1
                sum_chain_length = 0

            return {
                "num_source_variables": num_source_variables,
                "num_target_variables": num_target_variables,
                "max_chain_length": max_chain_length,
                "sum_chain_length": sum_chain_length,
                "chain_strength": chain_strength,
                "chain_break_method": chain_break_method,
            }
        else:
            return {
                "num_source_variables": 0,
                "num_target_variables": 0,
                "max_chain_length": 0,
                "sum_chain_length": 0,
                "chain_strength": 0,
                "chain_break_method": '',
            }

    def get_record_result(self):
        from statistics import mean

        decoded_samples = self.model.decode_sampleset(self.sample_set)
        best_result_raw = min(decoded_samples, key=lambda x: x.energy)
        mean_energy_raw = mean(s.energy for s in decoded_samples)

        decoded_samples = self.model.decode_sampleset(self.sample_set_pp)
        best_result_pp = min(decoded_samples, key=lambda x: x.energy)
        mean_energy_pp = mean(s.energy for s in decoded_samples)

        one = {
            "solver": "dwave",
            "machine": self.machine,
            "best_sample": best_result_raw.sample,
            "best_energy": best_result_raw.energy,
            "mean_energy": mean_energy_raw,
            "best_sample_pp": best_result_pp.sample,
            "best_energy_pp": best_result_pp.energy,
            "mean_energy_pp": mean_energy_pp
        }
        two = self.get_stats()
        return {**one, **two}
