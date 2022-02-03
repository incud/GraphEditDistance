import warnings

from qiskit.circuit.library import EfficientSU2

warnings.filterwarnings("ignore")

import numpy as np
from experimentcreator import generate_hamiltonian

from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA, VQE
from qiskit.optimization import QuadraticProgram
from qiskit.algorithms.optimizers.cobyla import COBYLA

from qiskit import transpile

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


def run_qaoa_stats(experiment, queue, i, p):

    max_iter = 1000
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
    # RUN QAOA
    optimizer = COBYLA(maxiter=max_iter)
    simulator = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=2048)
    qaoa = QAOA(operator, quantum_instance=simulator, optimizer=optimizer, p=(p + 1))

    circ_abs = qaoa.construct_circuit(np.array([0.01] * qaoa.var_form.num_parameters))[0]
    circ = transpile(circ_abs, basis_gates=['cx', 'u3'])
    num_qubits = circ.num_qubits
    num_params = qaoa.var_form.num_parameters
    circ_depth = circ.depth()
    circ_size = circ.size()

    row = {
        "experiment": i,
        "v": experiment['vertices'],
        "qubits": num_qubits,
        "parameters": num_params,
        "depth": circ_depth,
        "size": circ_size
    }

    queue.put(row)


def run_vqe_stats(experiment, queue, i, p):

    max_iter = 1000
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
    # RUN QAOA
    optimizer = COBYLA(maxiter=max_iter)
    qi = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=2048)
    ansatz = EfficientSU2(operator.num_qubits, reps=p, entanglement='full')
    vqe = VQE(operator, ansatz, optimizer, quantum_instance=qi)

    circ_abs = vqe.construct_circuit(np.array([0.01] * vqe.var_form.num_parameters))[0]
    circ = transpile(circ_abs, basis_gates=['cx', 'u3'])
    num_qubits = circ.num_qubits
    num_params = vqe.var_form.num_parameters
    circ_depth = circ.depth()
    circ_size = circ.size()

    row = {
        "experiment": i,
        "v": experiment['vertices'],
        "qubits": num_qubits,
        "parameters": num_params,
        "depth": circ_depth,
        "size": circ_size
    }

    queue.put(row)
