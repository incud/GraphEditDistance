import warnings
warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe
from dwavesolutioncreator import generate_dwave_dataframe, run_dwave_2000_experiment, run_dwave_advantage_experiment, run_dwave_leap_experiment
from qaoasolutioncreator import generate_qaoa_dataframe, run_qaoa_p1, run_qaoa_p3, run_qaoa_p5
import multiprocessing
from datetime import datetime


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
    # genera i grafi
    GRAPHS_PATH = "data/graphs.pickle"
    graph_df = generate_graph_dataframe(GRAPHS_PATH)
    graph_df.to_pickle(GRAPHS_PATH)
    # genera gli esperimenti
    EXPERIMENTS_PATH = "data/experiments.pickle"
    experiments_df = generate_experiments_dataframe(EXPERIMENTS_PATH, graph_df)
    experiments_df.to_pickle(EXPERIMENTS_PATH)
    # choose a subset of experiments
    experiments_df = experiments_df.iloc[:3, :]
    # simulated annealing
    
    # dwave 2000
    DWAVE_2000_PATH = "data/dwave_2000_solutions.pickle"
    dwave_2000_df = generate_dwave_dataframe(DWAVE_2000_PATH)
    run_experiments(experiments_df, run_dwave_2000_experiment, dwave_2000_df, DWAVE_2000_PATH)
    # dwave advantage
    DWAVE_ADVANTAGE_PATH = "data/dwave_advantage_solutions.pickle"
    dwave_advantage_df = generate_dwave_dataframe(DWAVE_ADVANTAGE_PATH)
    run_experiments(experiments_df, run_dwave_advantage_experiment, dwave_advantage_df, DWAVE_ADVANTAGE_PATH)
    # dwave leap
    DWAVE_LEAP_PATH = "data/dwave_leap_solutions.pickle"
    dwave_leap_df = generate_dwave_dataframe(DWAVE_LEAP_PATH)
    run_experiments(experiments_df, run_dwave_leap_experiment, dwave_leap_df, DWAVE_LEAP_PATH)
    # qaoa
    QAOA_1_PATH = "data/qaoa_p1_solutions.pickle"
    QAOA_3_PATH = "data/qaoa_p3_solutions.pickle"
    QAOA_5_PATH = "data/qaoa_p5_solutions.pickle"
    qaoa_1_df = generate_qaoa_dataframe(QAOA_1_PATH)
    qaoa_3_df = generate_qaoa_dataframe(QAOA_3_PATH)
    qaoa_5_df = generate_qaoa_dataframe(QAOA_5_PATH)
    run_experiments(experiments_df, run_qaoa_p1, qaoa_1_df, QAOA_1_PATH)
    run_experiments(experiments_df, run_qaoa_p3, qaoa_3_df, QAOA_3_PATH)
    run_experiments(experiments_df, run_qaoa_p5, qaoa_5_df, QAOA_5_PATH)
