import pandas as pd
from setup import *
import matplotlib.pyplot as plt


def speedUp(paralTime_lld, totalTime_lld):
    """
    speedup(P) = T_1 / T_P
    :param paralTime_lld: the execution time when using more than one thread.
    :param totalTime_lld: the execution time when using only one thread.
    :return: speedup(P) is the max of how many times that parallel computing could be faster than single-thread performance.
    """
    speedup = totalTime_lld / paralTime_lld
    return speedup


def efficiency(paralTime_lld, totalTime_lld, nThr):
    """
    E(P) = T_1 / (T_P * P)
    :param paralTime_lld: (T_P) is the parallel time (when using threads).
    :param totalTime_lld: (T_1) is the serial time (P = 1).
    :param nThr: number of threads.
    :return: the efficiency depending on number of threads P.
    """
    nWorkers = nThr
    efficiency = totalTime_lld / (paralTime_lld * nWorkers)
    return efficiency


def cost(paralTime_lld, nThr):
    """
    C(P) = T_P * P
    :param paralTime_lld: (T_P) is the parallel time (when using threads).
    :param nThr: number of threads.
    :return: the cost depending on number of threads P.
    """
    return paralTime_lld * nThr


def evaluateSOM(T_l, N_jobs_l, verbose=True):
    """
    Evaluate SOM and print metrics to file
    :param T_l: list of execution times
    :return:
    """
    rows = list()
    print("==== EVALUATION ====")
    T_1 = T_l[0]
    for P in range(len(N_jobs_l)):
        T_P = T_l[P]
        su = speedUp(T_P, T_1)
        ef = efficiency(T_P, T_1, P + 1)
        co = cost(T_P, P + 1)
        rows.append([N_jobs_l[P], weight_size, epochs, grid_rows, grid_cols, n_inputs, T_1, T_P, su, ef, co])

        if verbose:
            print(["=" * 25])
            print("N_THREADS = ", N_jobs_l[P])
            print("T_1 = ", T_1)
            print("T_P = ", T_P)
            print("Speed up = ", su)
            print("Efficiency = ", ef)
            print("Cost = ", co)

    # convert the list into dataframe row
    data = pd.DataFrame(rows,
                        columns=['N_TH', 'WEIGHT_SIZE', 'EPOCHS', 'GRID_ROWS', 'GRID_COLS', 'N_INPUTS', 'T_1',
                                 'T_P', 'SPEEDUP', 'EFFICIENCY', 'COST'])
    data.to_csv('stats.csv', mode='a', index=False)


def saveTimersSOM(SomLayer, N):
    t_bmu_l, t_neig_l, t_adj_l, t_res_l = SomLayer.getTimers()
    times_df = pd.DataFrame()
    times_df['bmu'] = t_bmu_l
    times_df['neig'] = t_neig_l
    times_df['adj'] = t_adj_l
    times_df['res'] = t_res_l
    times_df.to_csv(str(N) + ".csv", index=False)