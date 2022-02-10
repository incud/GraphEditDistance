import warnings

import numpy as np

warnings.filterwarnings("ignore")

from os import path
import pandas as pd
from experimentcreator import generate_hamiltonian

from qiskit.circuit.library import EfficientSU2
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.algorithms import VQE
from qiskit.optimization import QuadraticProgram
from qiskit.algorithms.optimizers.cobyla import COBYLA
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.aqua.utils.run_circuits import find_regs_by_name
from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe

from qiskit.circuit.library import QAOAAnsatz
import seaborn
import matplotlib.pyplot as plt

def replace_variable_name(name, map_names):
    new_name = name.replace("][", "_").replace("[", "_").replace("]", "")
    map_names[name] = new_name
    return new_name


def vector_to_sample(vector, map_names, variables):
    inv_map = {v: k for k, v in map_names.items()}
    sample = {}
    for i in range(len(vector)):
        var = inv_map[variables[i]]
        sample[var] = int(vector[i])
    return sample


def run_variational(operator, param_vector, is_vqe=True, p=1):
    qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=2048)

    if is_vqe:
        qc = EfficientSU2(operator.num_qubits, reps=p, entanglement='full')
        bind_dict = {}
        i = 0
        for key in qc.parameters:
            bind_dict[key] = param_vector[i]
            i = i + 1

        qc = qc.assign_parameters(bind_dict)
    else:
        qc_list = QAOA(operator, quantum_instance=qi).construct_circuit(param_vector)
        qc = qc_list[0]

    c = ClassicalRegister(qc.width(), name='c')
    q = find_regs_by_name(qc, 'q')
    qc.add_register(c)
    qc.barrier(q)
    qc.measure(q, c)
    result = qi.execute(qc)
    counts = result.get_counts()
    most_frequent_binary_string = sample_most_likely(counts)
    return most_frequent_binary_string


def run_qaoa(operator, param_vector, p=1):
    optimizer = COBYLA(maxiter=0)
    qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=2048)
    qaoa = QAOA(operator, quantum_instance=qi, optimizer=optimizer, p=p, initial_point=param_vector)
    result = qaoa.compute_minimum_eigenvalue()
    most_frequent_binary_string = sample_most_likely(result.eigenstate)
    return most_frequent_binary_string


def run_vqe_landscape(experiment, N=1000, p=1):
    bqm = experiment["bqm"]
    hamiltonian = generate_hamiltonian(experiment["g1"], experiment["g2"], experiment["a"], experiment["b"])

    # ===========================================================
    # CREATE OPERATOR
    model = hamiltonian.compile()
    qubo, offset = model.to_qubo()
    # rename variables from x[0] to x_0
    map_names = {}
    qubo = {(replace_variable_name(x1, map_names), replace_variable_name(x2, map_names)): v
            for (x1, x2), v in qubo.items()}
    variables = sorted(map_names.values())
    # create DOcplex quadratic program
    cplex_program = QuadraticProgram()
    # add the variables
    for x in variables:
        cplex_program.binary_var(x)
    # add the QUBO problem to DOcplex
    cplex_program.minimize(quadratic=qubo, constant=offset)
    operator, offset = cplex_program.to_ising()

    # ===========================================================
    # RUN VQE p=1
    matrix = np.random.random((N, N))
    for i in range(N):
        alpha0 = (-np.pi) + 2 * np.pi * i / N
        for j in range(N):
            alpha1 = (-np.pi) + 2 * np.pi * j / N
            binary_vector = run_variational(operator, [alpha0, alpha1] * 32, is_vqe=True, p=1)
            sample = vector_to_sample(binary_vector, map_names, variables)
            energy = bqm.energy(sample)
            matrix[i][j] = energy

    return matrix


def run_qaoa_landscape(experiment, N=1000, p=1, pos=False):

    bqm = experiment["bqm"]
    hamiltonian = generate_hamiltonian(experiment["g1"], experiment["g2"], experiment["a"], experiment["b"])

    # ===========================================================
    # CREATE OPERATOR
    model = hamiltonian.compile()
    qubo, offset = model.to_qubo()
    # rename variables from x[0] to x_0
    map_names = {}
    qubo = {(replace_variable_name(x1, map_names), replace_variable_name(x2, map_names)): v
            for (x1, x2), v in qubo.items()}
    variables = sorted(map_names.values())
    # create DOcplex quadratic program
    cplex_program = QuadraticProgram()
    # add the variables
    for x in variables:
        cplex_program.binary_var(x)
    # add the QUBO problem to DOcplex
    cplex_program.minimize(quadratic=qubo, constant=offset)
    operator, offset = cplex_program.to_ising()

    # ===========================================================
    # RUN QAOA p=1
    matrix = np.random.random((N, N))
    for i in range(N):
        if pos:
            alpha0 = np.pi * i / N
        else:
            alpha0 = (-np.pi) + 2 * np.pi * i / N
        for j in range(N):
            if pos:
                alpha1 = np.pi * j / N
            else:
                alpha1 = (-np.pi) + 2 * np.pi * j / N
            binary_vector = run_variational(operator, [alpha0, alpha1], is_vqe=False, p=1)
            sample = vector_to_sample(binary_vector, map_names, variables)
            energy = bqm.energy(sample)
            matrix[i][j] = energy
            print(".", end="")
        print("\n", i, end="")

    return matrix


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# genera i grafi
GRAPHS_PATH = "data/graphs.pickle"
graph_df = generate_graph_dataframe(GRAPHS_PATH)
graph_df.to_pickle(GRAPHS_PATH)
# genera gli esperimenti
EXPERIMENTS_PATH = "data/experiments.pickle"
experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df)
experiments_df.to_pickle(EXPERIMENTS_PATH)
experiment15 = experiments_df.loc[15]
experiment16 = experiments_df.loc[16]
experiment17 = experiments_df.loc[17]
experiment04 = experiments_df.loc[4]
experiment24 = experiments_df.loc[24]
N = 256
# matrix15 = run_qaoa_landscape(experiment15, N=N)
# np.savetxt("variational_landscape/qaoa_p1_experiment15_N256_landscape.csv", matrix15, delimiter=",")
# matrix16 = run_qaoa_landscape(experiment16, N=N)
# np.savetxt("variational_landscape/qaoa_p1_experiment16_N256_landscape.csv", matrix15, delimiter=",")
# matrix17 = run_qaoa_landscape(experiment17, N=N)
# np.savetxt("variational_landscape/qaoa_p1_experiment17_N256_landscape.csv", matrix16, delimiter=",")
# matrix04 = run_qaoa_landscape(experiment04, N=N)
# np.savetxt("variational_landscape/qaoa_p1_experiment04_N256_landscape.csv", matrix04, delimiter=",")
# matrix24 = run_qaoa_landscape(experiment24, N=N)
# np.savetxt("variational_landscape/qaoa_p1_experiment24_N256_landscape.csv", matrix24, delimiter=",")
# plt.imshow(matrix, cmap='hot', interpolation='nearest')
matrix15pos = run_qaoa_landscape(experiment15, N=N, pos=True)
np.savetxt("variational_landscape/qaoa_p1_experiment15_N256_pos_landscape.csv", matrix15pos, delimiter=",")
ax = seaborn.heatmap(matrix15pos)
plt.show()
