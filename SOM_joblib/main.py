import time
from SomModelParallel import *
from metrics import *
from report import *
from setup import *

if __name__ == "__main__":
    # execution times list
    if training is True:

        for weight_size in weight_size_:
            for grid_cols in grid_cols_:
                for grid_rows in grid_rows_:
                    for n_inputs in n_inputs_:
                        T_l = []
                        for P in range(len(N_jobs_l)):
                            print("N_JOBS = ", N_jobs_l[P])
                            dataset = np.random.rand(n_inputs, weight_size)
                            t_zero = time.time()
                            A = SomLayerP(grid_rows, grid_cols, weight_size, learning_rate, N_jobs_l[P])
                            A.training_som(epochs, dataset)
                            t_two = time.time()
                            delta = t_two - t_zero
                            # save execution times for metrics calculation
                            T_l.append(microsec(delta))
                            # export timers to csv
                            # saveTimersSOM(A, N_jobs_l[P])
                            evaluateSOM(T_l, N_jobs_l, P, weight_size, epochs, grid_rows, grid_cols, n_inputs)

    if report is True:
        # reportSOM()
        reportSOMPlot()


