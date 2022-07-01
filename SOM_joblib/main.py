import time
import numpy as np
from SomModelParallel import *
from metrics import *
from setup import *

if __name__ == "__main__":
    # execution times list
    T_l = []

    for P in range(len(N_jobs_l)):
        print("N_JOBS = ", N_jobs_l[P])
        dataset = np.random.rand(n_inputs, weight_size)
        t_zero = time.time()
        A = SomLayerP(grid_rows, grid_cols, weight_size, learning_rate, N_jobs_l[P])
        A.training_som(epochs, dataset)
        t_two = time.time()
        # save execution times for metrics calculation
        T_l.append(t_two - t_zero)
        # export timers to csv
        saveTimersSOM(A, N_jobs_l[P])

    evaluateSOM(T_l, N_jobs_l)



