import numpy as np
import matplotlib.pyplot as plt
from importData import generate_diffuse_tether_trajectories
from model_utils import GenerationMode
import model
import bayesianTools
from utils import backup_before_save, arr_of_length_of_true_segments


def generateSynthTrajs(N_steps, N_particle, dt, T_stick, T_unstick, D, A):
    generation_mode = GenerationMode.DONT_FORCE  # keep this
    init_S = None  # random
    do_post_processing = False  # calculate some extra statistics, this takes a short while
    undersample_ratio = 0  # 0.1
    save_files = False
    is_parallel = False  # relevant only if do_post_processing==True
    model_params = model.pack_model_params(T_stick, T_unstick, D, A, dt)  # for convenience

    df = generate_diffuse_tether_trajectories(T_stick, T_unstick, D, A, dt, N_steps, N_particle, init_S,
                                              do_post_processing, undersample_ratio, save_files, generation_mode,
                                              is_parallel)
    return df, model_params


def create_heatmaps(df, model_params, create_D_A_heatmap=True, D_A_grid=np.logspace(-.5, .5, 21),
                    create_T1_T2_heatmap=True, T1_T2_grid=np.logspace(-1., 3., 21)):
    X_arr_list = bayesianTools.extract_X_arr_list_from_df(df)
    dt_list = bayesianTools.extract_dt_list_from_df(df)

    if create_D_A_heatmap:
        print("Creating heatmap for D, A:")
        D_arr = D_A_grid * model_params['D']
        A_arr = D_A_grid * model_params['A']
        L_mat_D_A = np.zeros([len(D_arr), len(A_arr)])
        for n, D_iter in enumerate(D_arr):
            for m, A_iter in enumerate(A_arr):
                L_mat_D_A[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list,
                                                                                 model_params['T_stick'],
                                                                                 model_params['T_unstick'],
                                                                                 D_iter, A_iter,
                                                                                 is_parallel=True)

                print([n, m])
        D_A_mat_save_string = "L_mat_D_A.npy"
        backup_before_save(D_A_mat_save_string)
        np.save(D_A_mat_save_string, [L_mat_D_A, D_arr, A_arr])

    if create_T1_T2_heatmap:
        print("Creating heatmap for T_stick,T_unstick:")
        T1_arr = T1_T2_grid * model_params['T_stick']
        T2_arr = T1_T2_grid * model_params['T_unstick']
        L_mat_T1_T2 = np.zeros([len(T1_arr), len(T2_arr)])
        for n, T1_iter in enumerate(T1_arr):
            for m, T2_iter in enumerate(T2_arr):
                L_mat_T1_T2[n, m] = bayesianTools.multiple_trajectories_likelihood(X_arr_list, dt_list, T1_iter,
                                                                                   T2_iter, model_params['D'],
                                                                                   model_params['A'],
                                                                                   is_parallel=True)

                print([n, m])
        T1_T2_mat_save_string = "L_mat_T1_T2.npy"
        backup_before_save(T1_T2_mat_save_string)
        np.save(T1_T2_mat_save_string, [L_mat_T1_T2, T1_arr, T2_arr])
    print("Done with heatmaps!")


def compare_true_and_viterbi_paths(df, model_params, max_particles=10, do_graphics=True):
    N_particle = len(df.particle.unique())

    if do_graphics and (N_particle > max_particles):
        particle_iter_array = np.sort(np.random.choice(df.particle.unique(), max_particles))
    else:
        particle_iter_array = df.particle.unique()

    if do_graphics:
        N_rows = int(np.ceil(N_particle / 2))
        fig, ax = plt.subplots(N_rows, 2)

    X_arr_list = bayesianTools.extract_X_arr_list_from_df(df)
    dt_list = bayesianTools.extract_dt_list_from_df(df)
    accuracy_arr = np.zeros(N_particle)

    for i, particle_id in enumerate(particle_iter_array):
        X_arr = X_arr_list[particle_id]
        N_steps = len(X_arr)
        S, XT, L_est = bayesianTools.viterbi_algorithm(X_arr, model_params['T_stick'], model_params['T_unstick'],
                                                       model_params['D'], model_params['A'], dt_list[i], True)
        # fullState = [[S[j], X_arr[j], XT[j]] for j in range(N_steps)]

        S_true = df[df.particle == particle_id].state.values
        XT_true = df[df.particle == particle_id][["x_tether", "y_tether"]].values

        # fullState_true = [[S_true[j], X_arr[j], XT_true[j]] for j in range(N_steps)]
        # L_tru = model.model_trajectory_log_probability(S_true, X_arr, model_params)
        accuracy_arr[i] = np.sum(
            (S_true == S) * (
                    (S == 0) + (S == 1) * np.prod(np.isclose(XT, XT_true, rtol=1e-05, atol=1e-08, equal_nan=True),
                                                  axis=1))) / float(
            N_steps)
        if do_graphics:
            n = i % N_rows
            m = int(i / N_rows)
            ax[n, m].plot(np.arange(N_steps), S_true, '-')
            ax[n, m].plot(np.arange(N_steps), S, '--')
            ax[n, m].set_title("Accuracy: {}%".format(np.round(100 * accuracy_arr[i], 1)))

    if do_graphics:
        plt.tight_layout()
        save_string_fig = "true_vitrebi_comparison.png"
        backup_before_save(save_string_fig)
        plt.savefig(save_string_fig, dpi=400)
    save_string_accuracy = "true_vitrebi_accuracy_arr.npy"
    backup_before_save(save_string_accuracy)
    np.save(save_string_accuracy, [accuracy_arr, particle_iter_array, df])


def find_est_T1_T2_as_function_of_guess_T1_T2(df, model_params):
    D = model_params['D']
    A = model_params['A']
    T1_true = model_params['T_stick']
    T2_true = model_params['T_unstick']

    particle_iter_array = df.particle.unique()

    X_arr_list = bayesianTools.extract_X_arr_list_from_df(df)
    dt_list = bayesianTools.extract_dt_list_from_df(df)

    for i, particle_id in enumerate(particle_iter_array):
        X_arr = X_arr_list[particle_id]
        N_steps = len(X_arr)
        S, XT, L_est = bayesianTools.viterbi_algorithm(X_arr, model_params['T_stick'], model_params['T_unstick'],
                                                       model_params['D'], model_params['A'], dt_list[i], True)

        dt = dt_list[i][0]
        T1_est = dt * arr_of_length_of_true_segments(1 - S)
        T2_est = dt * arr_of_length_of_true_segments(S)
