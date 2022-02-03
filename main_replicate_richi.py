import warnings
warnings.filterwarnings("ignore")

from graphcreator import generate_graph_dataframe
from experimentcreator import generate_experiments_dataframe, generate_experiments_dataframe_richi
from dwavesolutioncreator import * # generate_dwave_dataframe, run_dwave_2000_experiment, run_dwave_advantage_experiment, run_simulated_experiment, run_dwave_leap_experiment
import multiprocessing
import numpy as np
from datetime import datetime
import pandas


def calculate_relative_distance(sol_df, exp_df):
    return (sol_df['normalize_cost'] - exp_df['exact_distance']) / np.where(sol_df['normalize_cost'] > 0, sol_df['normalize_cost'], 1)


def run_experiments(the_experiments_df, solver_function, solution_df, solution_path):
    print("run_experiments: solution path is", solution_path)
    queue = multiprocessing.Queue()

    for i, experiment in the_experiments_df.iterrows():
        print(f"running experiments {i} having data ...", end="")
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
    ABS = [(1, 0.1)]
    experiment_exact_distance_df = pandas.read_pickle("data/experiments.pickle")
    experiments_df = generate_experiments_dataframe_richi("results_ric/experiments_01_ann20micros.pickle", graph_df, ABS, experiment_exact_distance_df)
    experiments_df['exact_distance'] = experiment_exact_distance_df[experiment_exact_distance_df['b'] == 0.1]['exact_distance'].to_numpy()
    experiments_df.to_pickle("results_ric/experiments_01_ann20micros.pickle")

    # simulated annealing
    SIM_PATH = "results_ric/sa_01_ann20micros.pickle"
    sim_df = generate_dwave_dataframe(SIM_PATH)
    run_experiments(experiments_df, run_simulated_experiment, sim_df, SIM_PATH)
    sim_df['relative_distance'] = calculate_relative_distance(sim_df, experiments_df)
    sim_df['v'] = experiments_df['vertices']

    # dwave 2000 FATTO
    DWAVE_2000_PATH = "results_ric/2000_01_ann20micros.pickle"
    dwave_2000_df = generate_dwave_dataframe(DWAVE_2000_PATH)
    run_experiments(experiments_df, run_dwave_2000_experiment, dwave_2000_df, DWAVE_2000_PATH)
    dwave_2000_df['relative_distance'] = calculate_relative_distance(dwave_2000_df, experiments_df)
    dwave_2000_df['v'] = experiments_df['vertices']

    # # dwave advantage
    DWAVE_ADVANTAGE_PATH = "results_ric/advantage_01_ann20micros.pickle"
    dwave_advantage_df = generate_dwave_dataframe(DWAVE_ADVANTAGE_PATH)
    run_experiments(experiments_df, run_dwave_advantage_experiment, dwave_advantage_df, DWAVE_ADVANTAGE_PATH)
    dwave_advantage_df['relative_distance'] = calculate_relative_distance(dwave_advantage_df, experiments_df)
    dwave_advantage_df['v'] = experiments_df['vertices']

    print("SIM_DF\n", sim_df.groupby(['v'])['relative_distance'].mean())
    print("2000\n", dwave_2000_df.groupby(['v'])['relative_distance'].mean())
    print("ADVANTAGE\n", dwave_advantage_df.groupby(['v'])['relative_distance'].mean())

    # # dwave leap
    # DWAVE_LEAP_PATH = "data/dwave_leap_solutions.pickle"
    # dwave_leap_df = generate_dwave_dataframe(DWAVE_LEAP_PATH)
    # run_experiments(experiments_df, run_dwave_leap_experiment, dwave_leap_df, DWAVE_LEAP_PATH)

    # DWAVE_2000_1_PATH = f"data/dwave_2000_solutions_LP.pickle"
    # dwave_1_df = generate_dwave_dataframe(DWAVE_2000_1_PATH)
    # run_experiments(experiments_df, run_dwave_2000_experiment_lp, dwave_1_df, DWAVE_2000_1_PATH)

    # DWAVE_2000_2_PATH = f"data/dwave_2000_solutions_PM.pickle"
    # dwave_2_df = generate_dwave_dataframe(DWAVE_2000_2_PATH)
    # run_experiments(experiments_df, run_dwave_2000_experiment_pm, dwave_2_df, DWAVE_2000_2_PATH)

    # DWAVE_ADV_1_PATH = f"data/dwave_advantage_solutions_LP.pickle"
    # dwave_adv_1_df = generate_dwave_dataframe(DWAVE_ADV_1_PATH)
    # run_experiments(experiments_df, run_dwave_advantage_experiment_lp, dwave_adv_1_df, DWAVE_ADV_1_PATH)

    # DWAVE_ADV_2_PATH = f"data/dwave_advantage_solutions_PM.pickle"
    # dwave_adv_2_df = generate_dwave_dataframe(DWAVE_ADV_2_PATH)
    # run_experiments(experiments_df, run_dwave_advantage_experiment_pm, dwave_adv_2_df, DWAVE_ADV_2_PATH)

    # qaoa
    # QAOA_1_PATH = "data/qaoa_p1_solutions.pickle"
    # QAOA_3_PATH = "data/qaoa_p3_solutions.pickle"
    # QAOA_5_PATH = "data/qaoa_p5_solutions.pickle"
    # qaoa_1_df = generate_qaoa_dataframe(QAOA_1_PATH)
    # qaoa_3_df = generate_qaoa_dataframe(QAOA_3_PATH)
    # qaoa_5_df = generate_qaoa_dataframe(QAOA_5_PATH)
    # run_experiments(experiments_df, run_qaoa_p1, qaoa_1_df, QAOA_1_PATH)
    # run_experiments(experiments_df, run_qaoa_p3, qaoa_3_df, QAOA_3_PATH)
    # run_experiments(experiments_df, run_qaoa_p5, qaoa_5_df, QAOA_5_PATH)

    # VQE
    VQE_1_PATH = "data/vqe_p1_solutions.pickle"
    VQE_3_PATH = "data/vqe_p3_solutions.pickle"
    VQE_5_PATH = "data/vqe_p5_solutions.pickle"
    #vqe_1_df = generate_qaoa_dataframe(VQE_1_PATH)
    #vqe_3_df = generate_qaoa_dataframe(VQE_3_PATH)
    #vqe_5_df = generate_qaoa_dataframe(VQE_5_PATH)
    #run_experiments(experiments_df, run_qaoa_p1, vqe_1_df, VQE_1_PATH)
    #run_experiments(experiments_df, run_qaoa_p3, vqe_3_df, VQE_3_PATH)
    #run_experiments(experiments_df, run_qaoa_p5, vqe_5_df, VQE_5_PATH)
