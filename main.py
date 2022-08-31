from importData import import_all, generate_diffuse_tether_trajectories
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bayesianTools
from model_utils import GenerationMode
import time
import model
import tests
if __name__ == '__main__':
    N_steps = 1000
    N_particle = 10

    dt = 1. / 30
    T_stick = 200 * dt
    T_unstick = 200 * dt
    D = 0.3333
    A = D * dt / 5

    df, model_params = tests.generateSynthTrajs(N_steps, N_particle, dt, T_stick, T_unstick, D, A)
    tests.compare_true_and_viterbi_paths(df, model_params)

    # df = import_all(get_latest=False, is_parallel=True, break_trajectory_at_collisions=True, assign_traj_states=True)
    # N_steps = 1000
    # N_particle = 100
    #
    # dt = 1. / 30
    # T_stick = 200 * dt
    # T_unstick = 200 * dt
    # D = 0.3333
    # A = D * dt / 10
    #
    # generation_mode = GenerationMode.DONT_FORCE  # keep this
    # init_S = None  # random
    # do_post_processing = False  # calculate some extra statistics, this takes a short while
    # undersample_ratio = 0  # 0.1
    # save_files = False
    # is_parallel = False  # relevant only if do_post_processing==True
    #
    # df = generate_diffuse_tether_trajectories(T_stick, T_unstick, D, A, dt, N_steps, N_particle, init_S,
    #                                           do_post_processing, undersample_ratio, save_files, generation_mode,
    #                                           is_parallel)
    #
    # X_arr_list = bayesianTools.extract_X_arr_list_from_df(df)
    # dt_list = bayesianTools.extract_dt_list_from_df(df)
    #
    # D_arr = np.logspace(-.5, .5, 21) * D
    # A_arr = np.logspace(-.5, .5, 21) * A
    # L_mat = np.zeros([len(D_arr), len(A_arr)])
    # for n, D_iter in enumerate(D_arr):
    #     for m, A_iter in enumerate(A_arr):
    #         L_mat[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T_stick, T_unstick,
    #                                                                      D_iter, A_iter,
    #                                                                      is_parallel=True)
    #
    #         print([n, m])
    # np.save("L_mat", L_mat)
    #
    # T1_arr = np.logspace(-1., 3., 21) * T_stick
    # T2_arr = np.logspace(-1., 3., 21) * T_unstick
    # L_mat2 = np.zeros([len(T1_arr), len(T2_arr)])
    # for n, T1_iter in enumerate(T1_arr):
    #     for m, T2_iter in enumerate(T2_arr):
    #         L_mat2[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T1_iter, T2_iter, D, A,
    #                                                                       is_parallel=True)
    #
    #         print([n, m])
    # np.save("L_mat2", L_mat2)
    #
    # T1_arr = np.linspace(0.3, 1.7, 21) * T_stick
    # T2_arr = np.linspace(0.3, 1.7, 21) * T_unstick
    # L_mat3 = np.zeros([len(T1_arr), len(T2_arr)])
    # for n, T1_iter in enumerate(T1_arr):
    #     for m, T2_iter in enumerate(T2_arr):
    #         L_mat3[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T1_iter, T2_iter, D, A,
    #                                                                       is_parallel=True)
    #
    #         print([n, m])
    # np.save("L_mat3", L_mat3)
    #
    #
    # N_steps = 1000
    # N_particle = 100
    #
    # dt = 1. / 30
    # T_stick = 200 * dt
    # T_unstick = 200 * dt
    # D = 0.3333
    # A = D * dt
    #
    # generation_mode = GenerationMode.DONT_FORCE  # keep this
    # init_S = None  # random
    # do_post_processing = False  # calculate some extra statistics, this takes a short while
    # undersample_ratio = 0  # 0.1
    # save_files = False
    # is_parallel = False  # relevant only if do_post_processing==True
    #
    # df = generate_diffuse_tether_trajectories(T_stick, T_unstick, D, A, dt, N_steps, N_particle, init_S,
    #                                           do_post_processing, undersample_ratio, save_files, generation_mode,
    #                                           is_parallel)
    #
    # X_arr_list = bayesianTools.extract_X_arr_list_from_df(df)
    # dt_list = bayesianTools.extract_dt_list_from_df(df)
    #
    # D_arr = np.logspace(-.5, .5, 21) * D
    # A_arr = np.logspace(-.5, .5, 21) * A
    # L_mat = np.zeros([len(D_arr), len(A_arr)])
    # for n, D_iter in enumerate(D_arr):
    #     for m, A_iter in enumerate(A_arr):
    #         L_mat[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T_stick, T_unstick,
    #                                                                      D_iter, A_iter,
    #                                                                      is_parallel=True)
    #
    #         print([n, m])
    # np.save("L_hard_mat", L_mat)
    #
    # T1_arr = np.logspace(-1., 3., 21) * T_stick
    # T2_arr = np.logspace(-1., 3., 21) * T_unstick
    # L_mat2 = np.zeros([len(T1_arr), len(T2_arr)])
    # for n, T1_iter in enumerate(T1_arr):
    #     for m, T2_iter in enumerate(T2_arr):
    #         L_mat2[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T1_iter, T2_iter, D, A,
    #                                                                       is_parallel=True)
    #
    #         print([n, m])
    # np.save("L_hard_mat2", L_mat2)
    #
    # T1_arr = np.linspace(0.3, 1.7, 21) * T_stick
    # T2_arr = np.linspace(0.3, 1.7, 21) * T_unstick
    # L_mat3 = np.zeros([len(T1_arr), len(T2_arr)])
    # for n, T1_iter in enumerate(T1_arr):
    #     for m, T2_iter in enumerate(T2_arr):
    #         L_mat3[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T1_iter, T2_iter, D, A,
    #                                                                       is_parallel=True)
    #
    #         print([n, m])
    # np.save("L_hard_mat3", L_mat3)
