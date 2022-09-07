import numpy as np
import bayesianTools
import tests

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

    x_true = np.asarray([T_stick, T_unstick, D, A])

    x0_arr = [np.asarray([T_stick * 5, T_unstick / 4, D * 2, A / 2]),
              np.asarray([T_stick / 8, T_unstick * 3, D * .20, A * 25]),
              np.asarray([T_stick / 2, T_unstick * 2, D, A]),
              np.asarray([T_stick * 20, T_unstick, D, A * 5])]
    for x0 in x0_arr:
        tests.test_em_viterbi(x0, x_true, X_arr_list, dt_list, is_parallel=True,save_files=True)
