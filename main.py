import warnings
warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
from dwavesolutioncreator import generate_dwave_dataframe, run_dwave_2000_experiment, run_dwave_advantage_experiment, run_simulated_experiment, run_dwave_leap_experiment
from qaoasolutioncreator import generate_qaoa_dataframe, run_qaoa_p1, run_qaoa_p3, run_qaoa_p5
import multiprocessing
from datetime import datetime
import pandas

def run_experiments(the_experiments_df, solver_function, solution_df, solution_path):
    print("run_experiments: solution path is", solution_path)
    queue = multiprocessing.Queue()

    for i, experiment in the_experiments_df.iterrows():
        print(f"running experiments {i} having data {experiment.to_dict()} ...", end="")
        if i in list(solution_df["experiment"]):
            print("experiment already present")
            continue

        # fai partire il simulatore in un nuovo processo (pulizia memoria utilizzata ottimale)
        p = multiprocessing.Process(target=solver_function, args=(experiment, queue))
        start = datetime.now()
        p.start()
        p.join()
        end = datetime.now()
        result = queue.get()

        # salva il risultato sulla tabella
        solution_df.loc[len(solution_df)] = {"experiment": i, "start": start, "end": end, **result}

        # fai il backup
        solution_df.to_pickle(solution_path)

        print("|")

    # alla fine salva tutto
    solution_df.to_pickle(solution_path)


if __name__ == '__main__':
    pandas.set_option('display.max_rows', 500)
    pandas.set_option('display.max_columns', 500)
    pandas.set_option('display.width', 1000)
    # genera i grafi
    GRAPHS_PATH = "data/graphs.pickle"
    graph_df = generate_graph_dataframe(GRAPHS_PATH)
    graph_df.to_pickle(GRAPHS_PATH)
    # genera gli esperimenti
    EXPERIMENTS_PATH = "data/experiments.pickle"
    experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df)
    experiments_df.to_pickle(EXPERIMENTS_PATH)
    # choose a subset of experiments
    experiments_df = experiments_df.iloc[:27, :]
    # simulated annealing
    SIM_PATH = "data/simulated_annealing_solutions.pickle"
    sim_df = generate_dwave_dataframe(SIM_PATH)
    run_experiments(experiments_df, run_simulated_experiment, sim_df, SIM_PATH)

    from experimentcreator import *
    print("----------------------------TEST")
    H = generate_hamiltonian(graph_df.loc[0]['g'], graph_df.loc[2]['g'], 1, 0.1)
    bqm = H.compile().to_bqm()
    e = bqm.energy(sim_df.loc[6]['best_sample'])
    print(experiments_df.loc[6]['exact_distance'], e/.1)

    #print(experiments_df.head(27))
    #print(sim_df[['best_energy','best_energy_by_sample']])

    #sol= experiments_df.iloc[:3, -1]
    #print(sol)
    #sol_sim = sim_df.iloc[:3, 4]
    #print(sol_sim)
    #
    ##Test Index e grafi:
    #
    #g1_index, g2_index = experiments_df.iloc[0]["g1_index"], experiments_df.iloc[0]["g2_index"]
    #print(f"Index g1:{g1_index} g2:{g2_index}")
    #g1_index, g2_index = experiments_df.iloc[1]["g1_index"], experiments_df.iloc[1]["g2_index"]
    #print(f"Index g1:{g1_index} g2:{g2_index}")
    #g1_index, g2_index = experiments_df.iloc[2]["g1_index"], experiments_df.iloc[2]["g2_index"]
    #print(f"Index g1:{g1_index} g2:{g2_index}")
    #
    #g1, g2 = experiments_df.iloc[0]["g1"], experiments_df.iloc[0]["g2"]
    #print(f"ESP 1: G1 nodes: {g1.nodes()} G2 nodes: {g2.nodes()}, G1 edges: {g1.edges()} G2 edges: {g2.edges()}")
    #g1, g2 = experiments_df.iloc[1]["g1"], experiments_df.iloc[1]["g2"]
    #print(f"ESP 2: G1 nodes: {g1.nodes()} G2 nodes: {g2.nodes()}, G1 edges: {g1.edges()} G2 edges: {g2.edges()}")
    #g1, g2 = experiments_df.iloc[2]["g1"], experiments_df.iloc[2]["g2"]
    #print(f"ESP 3: G1 nodes: {g1.nodes()} G2 nodes: {g2.nodes()}, G1 edges: {g1.edges()} G2 edges: {g2.edges()}")


    ## dwave 2000
    #DWAVE_2000_PATH = "data/dwave_2000_solutions.pickle"
    #dwave_2000_df = generate_dwave_dataframe(DWAVE_2000_PATH)
    #run_experiments(experiments_df, run_dwave_2000_experiment, dwave_2000_df, DWAVE_2000_PATH)
    ## dwave advantage
    #DWAVE_ADVANTAGE_PATH = "data/dwave_advantage_solutions.pickle"
    #dwave_advantage_df = generate_dwave_dataframe(DWAVE_ADVANTAGE_PATH)
    #run_experiments(experiments_df, run_dwave_advantage_experiment, dwave_advantage_df, DWAVE_ADVANTAGE_PATH)
    ## dwave leap
    #DWAVE_LEAP_PATH = "data/dwave_leap_solutions.pickle"
    #dwave_leap_df = generate_dwave_dataframe(DWAVE_LEAP_PATH)
    #run_experiments(experiments_df, run_dwave_leap_experiment, dwave_leap_df, DWAVE_LEAP_PATH)
    ## qaoa
    #QAOA_1_PATH = "data/qaoa_p1_solutions.pickle"
    #QAOA_3_PATH = "data/qaoa_p3_solutions.pickle"
    #QAOA_5_PATH = "data/qaoa_p5_solutions.pickle"
    #qaoa_1_df = generate_qaoa_dataframe(QAOA_1_PATH)
    #qaoa_3_df = generate_qaoa_dataframe(QAOA_3_PATH)
    #qaoa_5_df = generate_qaoa_dataframe(QAOA_5_PATH)
    #run_experiments(experiments_df, run_qaoa_p1, qaoa_1_df, QAOA_1_PATH)
    #run_experiments(experiments_df, run_qaoa_p3, qaoa_3_df, QAOA_3_PATH)
    #run_experiments(experiments_df, run_qaoa_p5, qaoa_5_df, QAOA_5_PATH)
