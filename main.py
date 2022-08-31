from importData import import_all, generate_diffuse_tether_trajectories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bayesianTools
from model_utils import GenerationMode
import time
import model
import tests
import scipy

if __name__ == '__main__':
    N_steps = 1000
    N_particle = 40

    dt = 1. / 30
    T_stick = 200 * dt
    T_unstick = 200 * dt
    D = 0.3333
    A = D * dt / 2

    df, model_params = tests.generateSynthTrajs(N_steps, N_particle, dt, T_stick, T_unstick, D, A)

    X_arr_list = bayesianTools.extract_X_arr_list_from_df(df)
    dt_list = bayesianTools.extract_dt_list_from_df(df)


    def costFun(params):
        TT_stick, TT_unstick, DD, AA = params
        #     DD,AA = params
        #     TT_stick = T_stick
        #     TT_unstick = T_unstick
        t = time.time()
        L = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, TT_stick, TT_unstick, DD, AA,
                                                           is_parallel=True)
        print("Params=[{},{},{},{}]; L={}; In {} sec".format(
            np.round(TT_stick, 0), np.round(TT_unstick, 0), np.round(DD, 9), np.round(AA, 9), np.round(L, 9),
            np.round(time.time() - t, 1)))
        return -L


    res = scipy.optimize.minimize(costFun, [T_stick * 5, T_unstick / 4, D * 2, A / 2])
    np.save("results.npy", res)
