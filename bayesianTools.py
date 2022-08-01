import numpy as np
import pandas as pd
import itertools
# import autograd.numpy as np
# from autograd import grad, jacobian, hessian
# from autograd.extend import primitive, defvjp
# import scipy.optimize
import time
# import warnings
# import os
from extras import backupBeforeSave, isnotebook, isempty
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.special import factorial, gammaln, comb, iv
from config import OUTPUTS_DIR, MAX_PROCESSORS_LIMIT
from config import BAYESIAN_MAX_K_EVENTS, BAYESIAN_MIN_STATE_DURATION_FRAMES, MAX_K_TO_KEEP, \
    FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO
from extras import generate_CTMC_Matrix


# Trajectory labels:
# 0 = free
# 1 = stuck
# 0.5. - boundary
# each segment starts and ends with a 0.5 (boundaries are shared between segments).
# stuck segment takes all stuck points (but because it is some kind of variance, it's likelihood approximately a sum of n-1 gaussians)
# free segment takes all free points and the likelihood is the sum of differences (i.e. also n-1 gaussians).
# thanks to double counting of boundaries, for a total of N points, all segmentations will be effectively a sum of N-1 gaussians.


def generate_G_grid(X_arr, rho):
    X_stacked = np.vstack(X_arr)
    minX = np.min(X_stacked, axis=0)
    maxX = np.max(X_stacked, axis=0)
    stdX = np.std(X_stacked, axis=0)
    x_min = minX[0] - 0.01 * stdX[0]
    x_max = maxX[0] + 0.01 * stdX[0]
    y_min = minX[1] - 0.01 * stdX[1]
    y_max = maxX[1] + 0.01 * stdX[1]

    x_arr = np.linspace(x_min, x_max, int(np.sqrt(rho) * (x_max - x_min)))
    y_arr = np.linspace(x_min, x_max, int(np.sqrt(rho) * (y_max - y_min)))

    G = len(x_arr) * len(y_arr)
    [X_grid, Y_grid] = np.meshgrid(x_arr, y_arr)
    return X_grid, Y_grid, G


def tryViterbi(X_arr, params, dt):
    logProcess = False
    N = len(X_arr)
    K = N + 1
    T_stick, T_unstick, D, A = params
    forgetOldStuckThreshold = FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO * A

    # There are K=N+1 hidden states:
    # 0 is free, and i is got stuck at the transition i-2 -> i-1 at X[i-1] (i>=1)
    # This means at time n there are only n+2 possible states (0<=n<= N-1)

    # Likelihood matrix L_mat[i,n] is the likelihood of the most likely path that
    # reaches state i at time n (n time points so this is the sum of n-1 step contributions + initial condition)
    L_mat = np.zeros([K, N], dtype=float) - np.inf

    # State matrix S_mat[i,n] is the state at time n-1 which from which the most likely path leading to state i at time n-1
    S_mat = np.zeros([K, N], dtype=int) - 1

    # We calculate this matrix now instead of each step likelihood calculation:
    P = generate_CTMC_Matrix(T_stick, T_unstick, dt)

    def L(n, k, i):  # transition log likelihood from hidden state k at time n to hidden state i at time n+1
        # Parse n,k,i to cur_state and next_state:
        if k == 0:
            cur_state = [0, X_arr[n], [0, 0]]
        else:
            cur_state = [1, X_arr[n], X_arr[k - 1]]

        if i == 0:
            next_state = [0, X_arr[n + 1], [0, 0]]
        else:
            next_state = [1, X_arr[n + 1], X_arr[i - 1]]
        return stepLogLikelihood(cur_state, next_state, dt, params, enforceSameTether=False, P=P)

    # Initialize: equal prior for stuck or free at time 0
    L_mat[0, 0] = 0.
    L_mat[1, 0] = 0.
    # This is not really for a prior but so we can add likelihoods
    L_mat[:, :] = 0.
    L_mat[:, N - 1] = -np.inf  # all unreached destinations are by default with 0 probability

    valid_K_range = np.zeros(K, dtype=bool)
    valid_K_range[0] = True
    valid_source_state_list = np.where(valid_K_range)[0]

    isNotebook = isnotebook()
    logL_paths_to_i = np.zeros(MAX_K_TO_KEEP) - np.inf
    for j in range(1, N):  # time - we follow the Wikipedia Viterbi entry with indices j,i,k
        if logProcess:
            if j % int(np.ceil(N / 10)) == 0:
                if isNotebook:
                    print("Processed {}% of the trajectory".format(int(np.floor(100 * j / N))), end="\r")
                else:
                    print("Processed {}% of the trajectory".format(int(np.floor(100 * j / N))))

        # Since the number of possible states (K) grows each time step (stuck at X[0], stuck at X[1], etc.) we discard
        # some states according to the following criteria:

        # 1. If X[m] is very far from X[j], there is no point to consider the "(still) stuck at X[m]" state.
        for m in valid_source_state_list:
            if (m > 0) & (np.sum((X_arr[j] - X_arr[m - 1]) ** 2) > forgetOldStuckThreshold):
                # print("At time {}, discarded the stuck at {} option which was {}A apart ".format(j, m-1, int(((np.sum((X_arr[j] - X_arr[m - 1]) ** 2)))/A)))
                valid_K_range[m] = False

        # 2. If the particle is stuck for a long time, the above criterion won't discard any states. however, getting
        # stuck at the right time is still with quite a higher likelihood than staying free for some more time then
        # sticking later. so out of all the stuck states, we keep only the MAX_K_TO_KEEP most likely ones (and always
        # keeping free (k=0) and just stuck (k=j)), knowing we are not really missing any high likelihood global paths.
        valid_source_state_list = np.where(valid_K_range)[0]
        if len(valid_source_state_list[1:]) > MAX_K_TO_KEEP - 2:  # minus 2 because of state 0 and j
            valid_K_range[valid_source_state_list[1:][np.argsort(L_mat[valid_source_state_list[1:], j - 1])[:-(MAX_K_TO_KEEP - 2)]]] = False

        valid_K_range[j] = True
        valid_source_state_list = np.where(valid_K_range)[0]
        logL_paths_to_i -= np.inf
        for i in valid_source_state_list:  # i is the destination state.
            if i == 0:  # only the free state has multiple paths leading to it (more than one source state)
                # removed array construction because it's a bit slow (was: logL_paths_to_i = [L_mat[k, j - 1] + L(j - 1, k, i) for k in valid_K_list]  # source states)
                for ind, k in enumerate(valid_source_state_list):
                    logL_paths_to_i[ind] = L_mat[k, j - 1] + L(j - 1, k, i)  # source states
                ind_max = np.argmax(logL_paths_to_i)
                S_mat[i, j] = valid_source_state_list[ind_max]
                L_mat[i, j] = logL_paths_to_i[ind_max]
            else:
                # for 0<i<j+1, can only get to i from i (stuck in the same place)
                # as long as this i is still valid (very far tether points are forgotten)
                S_mat[i, j] = i
                L_mat[i, j] = L_mat[i, j - 1] + L(j - 1, i, i)
        #In addition consider the sticking from free to stuck at location j (state i=j+1)
        S_mat[j+1, j] = 0
        L_mat[j+1, j] = L_mat[0, j - 1] + L(j - 1, 0, i)

    # Now follow back to the start to find the path itself:
    viterbiPath = np.zeros(N, dtype=int)
    viterbiPath[N - 1] = np.argmax(L_mat[:, N - 1])  # this indicates the best path
    for j in reversed(range(1, N)):
        viterbiPath[j - 1] = S_mat[viterbiPath[j], j]

    # Now convert this to S and X_tether:
    S_arr = np.zeros(N, dtype=int)
    X_tether_arr = np.zeros([N, 2], dtype=float)

    for j in range(N):
        st = viterbiPath[j]  # state
        if st == 0:
            S_arr[j] = 0
            X_tether_arr[j, :] = np.nan
        else:
            S_arr[j] = 1
            X_tether_arr[j, :] = X_arr[st - 1]

    if logProcess:
        if isNotebook:
            print("")
        print("Done with Viterbi Algorithm!")
    return S_arr, X_tether_arr  # , S_mat, L_mat, viterbiPath

    #
    # m = 10
    # M = m ** 2
    # K = M + 1
    # A = params[3]
    # rho = M / (2 * A)
    # G_grid = generate_G_grid(X_arr, rho)
    #
    # # Now the hidden state is given by s (0 or 1), and i,j of G_grid
    #
    # zero_A_grid = generate_A_grid([0, 0], params[3], M)  # tether point grid centered around zero
    #
    # # States are encoded by [s,x_tether,y_tether] where s is 0 or 1
    #
    # genPossibleHiddenStatesArr = lambda X: \
    #     np.append(np.asarray(list(itertools.product(1, zero_grid + X))), [[0, [0., 0.]]], axis=0)
    #
    # # for s in [0,1]:
    # #     if s==0:
    # #         L_mat[0,0] =
    # #     else:
    # #         for i,X_tether in enumerate(X_arr[0]+zero_grid):
    #
    # for n in range(1, N):
    #     for s in [0, 1]:
    #         if s == 0:
    #             pass
    #         else:
    #             for i, X_tether in enumerate(X_arr[n] + zero_grid):
    #                 pass
    #


def generate_A_grid(X, A, M):
    L = 2 * np.sqrt(A)  # square side length
    xx = np.linspace(-L, L, int(np.sqrt(M))) + X[0]
    yy = np.linspace(-L, L, int(np.sqrt(M))) + X[1]
    grid = np.asarray(list(itertools.product(xx, yy)))
    return grid


def stepLogLikelihood(cur_state, next_state, dt, params, enforceSameTether=False, P=[]):
    [T_stick, T_unstick, D, A] = params
    [S_cur, X_cur, X_tether_cur] = cur_state
    [S_next, X_next, X_tether_next] = next_state

    # First enforce the tether to stay the same unless this is a sticking event or the particle is not stuck
    # Doing this enforcement slows the function, and this enforcement can be ensured at the iterative Viterbi algorithm
    # [don't link states tethered to Y0 with a state tethered to Y1]
    if enforceSameTether:
        isTetherNotSame = (((S_next == 1.) & (S_cur == 1.))) & (~np.all(np.isclose(X_tether_cur, X_tether_next)))
    else:
        isTetherNotSame = False

    if isTetherNotSame:
        L = -np.inf
    else:
        L = 0.
        if isempty(P):
            P = generate_CTMC_Matrix(T_stick, T_unstick, dt)

        # Temporal part (state transitions)
        L += np.log(P[int(S_cur), int(S_next)])
        # Spatial part
        if S_cur == 0:  # free
            L += -(np.log(4 * np.pi * D * dt) + np.sum(X_next - X_cur) ** 2 / (4 * D * dt))
            # New tether for sticking
            if S_next == 1:
                L += 0  # -(np.log(np.pi * A) + np.sum(X_next - X_tether_next) ** 2)
        else:
            L += -(np.log(np.pi * A) + np.sum(X_next - X_tether_cur) ** 2 / A)
    return L


def labeledParticleLikelihood(trajData, ind_switches, init_state, enforceModelParameters=[], returnParameters=False):
    "Calculate the model likelihood + best-estimate parameters for a specific particle with a LABELED trajectory"
    # 1. This is the building block of all the likelihood functions which we optimize.
    # 2. This function incorporates everything about the model
    t, _, _, _, _ = trajData
    N = len(t)
    T = t[-1] - t[0]
    deltaT = T / N
    if np.isscalar(ind_switches):
        ind_switches = [ind_switches]
    ind_segment_borders = genBorders(ind_switches, N)

    k = len(ind_switches)
    if k % 2 == 0:
        final_state = init_state
    else:
        final_state = 1 - init_state

    mode1_segs = np.empty(1 + int(np.floor(k / 2)), dtype=object)
    mode2_segs = np.empty(int(np.ceil(k / 2)), dtype=object)
    # trajDataStacked = np.vstack(trajData)
    for i in range(len(ind_segment_borders) - 1):
        if i % 2 == 0:
            mode1_segs[int(i / 2)] = trajData[:, ind_segment_borders[i]:(ind_segment_borders[i + 1] + 1)]
        else:
            mode2_segs[int(i / 2)] = trajData[:, ind_segment_borders[i]:(ind_segment_borders[i + 1] + 1)]

    if init_state == 0:
        free_segs = mode1_segs
        stuck_segs = mode2_segs
        k_stick = len(stuck_segs)
        k_unstick = len(free_segs) - 1
    else:
        free_segs = mode2_segs
        stuck_segs = mode1_segs
        k_stick = len(stuck_segs) - 1
        k_unstick = len(free_segs)

    n_free = len(free_segs)
    n_stuck = len(stuck_segs)

    D_est = 0.
    N_free_steps = 0
    T_free_total = 0.
    sum_logT_spent_free = 0.
    last_dt_free = 0.
    sum_log_last_dt_free = 0.  # because of possibly uneven sampling

    for n_seg, seg in enumerate(free_segs):
        _, dt, dr2, _, _ = seg
        len_seg = seg.shape[1]
        D_est += np.nansum(dr2 / (4 * dt))
        T_free_total += np.nansum(dt)
        sum_logT_spent_free += np.nansum(np.log(dt))  # for later calc
        N_free_steps += len_seg
        if ~((final_state == 0) & (n_seg + 1 == n_free)):
            last_dt_free += dt[-1]
            sum_log_last_dt_free += np.log(dt[-1])

    if N_free_steps != 0:
        D_est = D_est / N_free_steps
        if k_stick > 0:
            T_stick_est = (T_free_total - last_dt_free) / k_stick
        else:
            T_stick_est = np.nan
    else:
        D_est = np.nan
        T_stick_est = np.nan

    A_est = 0.
    N_stuck_steps = 0
    T_stuck_total = 0.
    last_dt_stuck = 0.
    sum_log_last_dt_stuck = 0.  # because of possibly uneven sampling
    for n_seg, seg in enumerate(stuck_segs):
        _, dt, _, x, y = seg
        len_seg = seg.shape[1]
        A_est += len_seg * (np.nanvar(x) + np.nanvar(y))
        T_stuck_total += np.nansum(dt)
        N_stuck_steps += len_seg
        if ~((final_state == 1) & (n_seg + 1 == n_stuck)):
            last_dt_stuck += dt[-1]
            sum_log_last_dt_stuck += np.log(dt[-1])

    if N_stuck_steps != 0:
        A_est = A_est / N_stuck_steps
        if k_unstick > 0:
            T_unstick_est = (T_stuck_total - last_dt_stuck) / k_unstick
        else:
            T_unstick_est = np.nan
    else:
        A_est = np.nan
        T_unstick_est = np.nan

    # Now add option to force external parameters in the calculation of the likelihood
    est_params = [T_stick_est, T_unstick_est, D_est, A_est]
    params = est_params.copy()  # default
    if not isempty(enforceModelParameters):
        for j in range(4):
            if ~np.isnan(est_params[j]):
                params[j] = enforceModelParameters[j]
    [T_stick, T_unstick, D, A] = params

    logP_spatial_free = -N_free_steps * D_est / D - (N_free_steps * np.log(4 * np.pi * D) + sum_logT_spent_free)
    logP_spatial_stuck = -N_stuck_steps * (A_est / A + np.log(np.pi * A))

    if k_stick > 0:
        logP_temporal_free = -k_stick * T_stick_est / T_stick - k_stick * np.log(T_stick) + sum_log_last_dt_free
    else:
        logP_temporal_free = N_free_steps * T_free_total / T_stick
    if k_unstick > 0:
        logP_temporal_stuck = -k_unstick * T_unstick_est / T_unstick - k_unstick * np.log(
            T_unstick) + sum_log_last_dt_stuck
    else:
        logP_temporal_stuck = N_stuck_steps * T_stuck_total / T_unstick

    logP_spatial = np.nansum([logP_spatial_free, logP_spatial_stuck])
    logP_temporal = np.nansum([logP_temporal_free, logP_temporal_stuck])

    L = logP_spatial + logP_temporal
    if ~np.isfinite(L):
        print("ERROR! NAN LIKELIHOOD FIX THIS")
        print("L={}".format(L))

    if returnParameters:
        return L, est_params
    else:
        return L


def genBorders(ind_switches, N):
    "Utility function for adding trajectory boundaries to an array of state-switching indices"
    if np.isscalar(ind_switches):
        len_ind_switches = 1
    else:
        len_ind_switches = len(ind_switches)
    ind_borders = np.zeros(len_ind_switches + 2, dtype=int)
    ind_borders[1:-1] = ind_switches
    ind_borders[-1] = N - 1
    return ind_borders


def genStatesArr(ind_switches, N_steps, init_state):
    "Utility function to convert from switching times to a state array compatible with the dataframes"
    ind_borders = genBorders(ind_switches, N_steps)
    s = np.zeros(N_steps)
    cur_s = 0.
    for i in range(len(ind_borders) - 1):
        s[ind_borders[i]:ind_borders[i + 1]] = cur_s
        cur_s = 1 - cur_s
    s[ind_borders] = 0.5
    if init_state == 1:
        s = 1 - s
    return s


def labelAllTrajectories(df, isPar=False):
    "Label and estimate parameters for all the trajectories in the dataframe, video-wise"
    df_exp_arr = [df_exp for _, df_exp in df.groupby(["experiment", "filename"])]
    N_videos = len(df_exp_arr)
    modified_df_exp_arr = np.empty(N_videos, dtype=object)
    for i, df_exp in enumerate(df_exp_arr):
        print("Labeling trajectories for video {} / {}".format(1 + i, N_videos))
        T_characteristic = df_exp[df_exp.isMobile].groupby("particle").trajDurationSec.first().mean()
        states_arr, L_array, _, _, params_mat = labelExperimentTrajectories(df_exp, 0.5 * T_characteristic,
                                                                            0.5 * T_characteristic,
                                                                            isPar=isPar)
        df_exp["state"] = states_arr
        df_exp["bayesian_T_stick"] = params_mat[:, 0]
        df_exp["bayesian_T_unstick"] = params_mat[:, 1]
        df_exp["bayesian_D"] = params_mat[:, 2]
        df_exp["bayesian_A"] = params_mat[:, 3]
        df_exp["bayesian_logLikelihood"] = L_array
        modified_df_exp_arr[i] = df_exp
    df = pd.concat(modified_df_exp_arr).sort_index()
    return df


def labelExperimentTrajectories(df, T_stick_guess, T_unstick_guess, isPar=False):
    "Label and estimate parameters for all the trajectories in an experiment/video"
    # This is an adaptive optimization scheme - first we label particle-wise, then we try to incorporate what we learned
    # particle-wise to better estimate the labels of all the particles together, then again re-estimate parameters
    # COMMENT 04/07/22: at the moment this is just particle-wise. 'learning' the parameters seems to fail
    particles_list = [p for p, _ in df.groupby("particle")]
    N_particles = len(particles_list)
    trajDuration_list = [len(particle_df) for _, particle_df in df.groupby("particle")]
    trajData_iterable = [np.vstack(extract_trajData_from_df(particle_df)) for _, particle_df in df.groupby("particle")]

    if np.isin('ismobile', [x.lower() for x in df.columns.to_numpy()]):
        isMobileList = [particle_df.isMobile.values[0] for _, particle_df in df.groupby("particle")]
    else:
        isMobileList = np.ones(N_particles).astype(bool)

    config_array = np.empty(N_particles, dtype=object)
    params_array = np.empty(N_particles, dtype=object)
    L_array = np.zeros(N_particles)

    N_reestimate_iterations = 0  # THINK WHAT TO DO ABOUT THIS

    # Step 1 - for each particle, estimate the most likely stuck/free configuration and derive estimated parameters from it
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for i, output in enumerate(executor.map(labelParticleTrajectory, trajData_iterable, repeat(T_stick_guess),
                                                    repeat(T_unstick_guess), isMobileList)):
                if isnotebook():
                    print("Labeling trajectory of particle {}/{}".format(1 + i, N_particles), end="\r")
                else:
                    print("Labeling trajectory of particle {}/{}".format(1 + i, N_particles))
                best_config = output[1]
                config_array[i] = best_config
                L_array[i], params_array[i] = labeledParticleLikelihood(trajData_iterable[i], best_config[1],
                                                                        best_config[0], returnParameters=True)
                if not isMobileList[i]:
                    L_array[i] = np.nan
                    params_array[i] = np.asarray([np.nan, np.nan, np.nan, params_array[i][3]])


    else:
        for i, trajData in enumerate(trajData_iterable):
            if isnotebook():
                print("Labeling trajectory of particle {}/{}".format(1 + i, N_particles), end="\r")
            else:
                print("Labeling trajectory of particle {}/{}".format(1 + i, N_particles))
            output = labelParticleTrajectory(trajData, T_stick_guess, T_unstick_guess, isMobileList[i])
            best_config = output[1]
            config_array[i] = best_config
            L_array[i], params_array[i] = labeledParticleLikelihood(trajData, best_config[1], best_config[0],
                                                                    returnParameters=True)
            if not isMobileList[i]:
                L_array[i] = np.nan
                params_array[i] = np.asarray([np.nan, np.nan, np.nan, params_array[i][3]])
    # Step 2 -re-estimate parameters
    params_stacked = np.vstack(params_array)
    # Note: here there might be numpy warning of mean/var of empty slice, if there was no estimate of one of the
    # parameters (e.g. all particles never unstuck so there is no T_unstick).
    params_mean = np.nanmean(params_stacked, axis=0)
    params_std = np.nanstd(params_stacked, axis=0)

    [T_stick, T_unstick, D, A] = params_mean

    for N_iter in range(N_reestimate_iterations):
        # Step 3* - for each particle, estimate the most likely stuck/free configuration given the mean parameters
        if isPar:
            with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
                for i, output in enumerate(
                        executor.map(labelParticleTrajectory, trajData_iterable, repeat(T_stick_guess),
                                     repeat(T_unstick_guess), isMobileList, repeat(params_mean))):
                    if isnotebook():
                        print("Labeling trajectory of particle {}/{}".format(1 + i, N_particles), end="\r")
                    else:
                        print("Labeling trajectory of particle {}/{}".format(1 + i, N_particles))
                    best_config = output[1]
                    config_array[i] = best_config
                    L_array[i], params_array[i] = labeledParticleLikelihood(trajData_iterable[i], best_config[1],
                                                                            best_config[0], returnParameters=True)
                    if not isMobileList[i]:
                        L_array[i] = np.nan
                        params_array[i] = np.asarray([np.nan, np.nan, np.nan, params_array[i][3]])

        else:
            for i, trajData in enumerate(trajData_iterable):
                output = labelParticleTrajectory(trajData, T_stick, T_unstick, isMobileList[i],
                                                 enforceModelParameters=params_mean)
                best_config = output[1]
                config_array[i] = best_config
                L_array[i], params_array[i] = labeledParticleLikelihood(trajData, best_config[1], best_config[0],
                                                                        returnParameters=True)
                if not isMobileList[i]:
                    L_array[i] = np.nan
                    params_array[i] = np.asarray([np.nan, np.nan, np.nan, params_array[i][3]])

        params_stacked = np.vstack(params_array)
        params_mean = np.nanmean(params_stacked, axis=0)
        params_std = np.nanstd(params_stacked, axis=0)
        [T_stick, T_unstick, D, A] = params_mean

    # Step 4 - create arrays for updating the df
    states_arr = np.hstack([genStatesArr(config_array[j][1], trajDuration_list[j], config_array[j][0]) for j in
                            range(N_particles)])
    params_mat = np.vstack([np.zeros([trajDuration_list[j], 1]) + params_array[j] for j in range(N_particles)])
    L_long_array = np.hstack([np.zeros([trajDuration_list[j]]) + L_array[j] for j in range(N_particles)])

    return [states_arr, L_long_array, params_mean, params_std, params_mat]


def labelParticleTrajectory(trajData, T_stick_guess, T_unstick_guess, isMobile=True, enforceModelParameters=[],
                            CG_factor=0.667):
    "Receive a particle trajectory and find the best labeing for it"
    # This function is basically a complicated optimization scheme for the state-switching times.
    # The maximal number of switching times is selected a priory from the Pk distribution, requiring a prior of the time parameters
    # CG_factor - between 0 and 1, it is a tradeoff between performance and accuracy. better not decrease it below 0.75.
    # closer to 1 is more coarse graining -> better performance but short events might be missed
    if not isMobile:
        best_L = np.nan
        best_config = [1., []]
    else:
        logMessages = False
        smooth_T_factor = 0.5

        t, _, _, _, _ = trajData
        N = len(t)
        T = t[-1] - t[0]

        T_stick_guess = T_stick_guess ** (1 - smooth_T_factor) * T ** (smooth_T_factor)
        T_unstick_guess = T_unstick_guess ** (1 - smooth_T_factor) * T ** (smooth_T_factor)

        K_range = np.arange(BAYESIAN_MAX_K_EVENTS + 1)

        calcLikelihood = lambda switch_time_indices, initial_state: labeledParticleLikelihood(trajData,
                                                                                              switch_time_indices,
                                                                                              initial_state,
                                                                                              enforceModelParameters=enforceModelParameters,
                                                                                              returnParameters=False)

        L_mat = np.empty(2, dtype=object)
        config_mat = np.empty(2, dtype=object)

        for i, init_state in enumerate([0., 1.]):
            Pk = kEventsProb(K_range, init_state, T, T_stick_guess, T_unstick_guess)
            max_K = K_range[np.min(np.where(np.cumsum(Pk) > 0.9), initial=np.max(K_range))]
            log2c = int(np.min([np.floor(CG_factor * np.log2(N)),
                                np.floor(np.log2(N / max_K))]))
            c = 2 ** log2c
            coarse_switch_time_inds = np.arange(c, N, step=c)
            # max_K = np.max(K_range[indLikelyEnoughPk(Pk)])
            # max_K = np.max([max_K, 2])  # allow up to K=2 always (TBD)
            # coarse graining factor
            # log2c = int(np.floor(CG_factor * np.log2(N)))
            # c = 2 ** log2c
            # max_K = np.min([max_K, len(coarse_switch_time_inds)])
            # max_K=6
            K_to_iterate = np.arange(0, max_K + 1, step=1)
            L_arr_K = np.zeros(len(K_to_iterate))
            config_arr_K = np.empty(len(K_to_iterate), dtype=object)
            if logMessages: print("K to iterate: {}".format(K_to_iterate))
            for j, K in enumerate(K_to_iterate):
                if K == 0:
                    new_switch_time_inds = []
                elif K == 1:  # don't coarse or fine grain
                    all_switch_time_inds = np.arange(BAYESIAN_MIN_STATE_DURATION_FRAMES,
                                                     N - BAYESIAN_MIN_STATE_DURATION_FRAMES + 1)
                    if not isempty(all_switch_time_inds):
                        L_arr_switch_times = np.zeros(len(all_switch_time_inds))
                        if logMessages: print(
                            "Iterating over {} configurations of switch times:".format(len(all_switch_time_inds)))
                        for n, switch_time_inds in enumerate(all_switch_time_inds):
                            L_arr_switch_times[n] = calcLikelihood(switch_time_inds, init_state)
                        new_switch_time_inds = all_switch_time_inds[np.argmax(L_arr_switch_times)]
                    else:
                        new_switch_time_inds = np.nan

                else:
                    # first find the best coarse-grained switch times:
                    all_switch_time_inds = list(itertools.combinations(coarse_switch_time_inds, K))
                    L_arr_switch_times = np.zeros(len(all_switch_time_inds))
                    if logMessages: print("Iterating over {} configurations of coarse-grained switch times:".format(
                        len(all_switch_time_inds)))
                    # print(all_switch_time_inds)
                    for n, switch_time_inds in enumerate(all_switch_time_inds):
                        L_arr_switch_times[n] = calcLikelihood(switch_time_inds, init_state)
                    max_ind = np.argmax(L_arr_switch_times)
                    best_coarse_time_inds = np.asarray(all_switch_time_inds[max_ind])

                    if logMessages: print("Best coarse time indices: {}".format(best_coarse_time_inds))
                    # best_coarse_time_inds = (np.arange(1,K+1)* N/(K+1+np.random.randint(2)*K/2)).astype(int)
                    # print(best_coarse_time_inds)
                    best_time_inds = best_coarse_time_inds * 1  # copy the array
                    cur_L = L_arr_switch_times[max_ind]
                    beta = 0.1
                    std_factor = 1
                    iter = 0
                    n_switches = 0
                    while beta < 1:
                        switch_inds = best_time_inds * 1
                        randK = np.random.randint(K)
                        if randK == 0:
                            min_d = 0 + BAYESIAN_MIN_STATE_DURATION_FRAMES - best_time_inds[randK]
                        else:
                            min_d = best_time_inds[randK - 1] + BAYESIAN_MIN_STATE_DURATION_FRAMES - best_time_inds[
                                randK]
                        if randK == K - 1:
                            max_d = N - BAYESIAN_MIN_STATE_DURATION_FRAMES - best_time_inds[randK]
                        else:
                            max_d = best_time_inds[randK + 1] - BAYESIAN_MIN_STATE_DURATION_FRAMES - best_time_inds[
                                randK]

                        d = 0
                        d_arr = []
                        if (min_d <= -1) & (max_d >= 1):
                            while (d == 0) | (d < min_d) | (d > max_d):
                                d = int(np.random.randn() * (max_d - min_d) / 2 * std_factor + (min_d + max_d) / 2)
                                d_arr.append(d)

                        if d != 0:
                            switch_inds[randK] += d
                            try_L = calcLikelihood(switch_inds, init_state)
                            # print(np.min([(try_L - cur_L) * beta, 0]))
                            if np.exp(np.min([(try_L - cur_L) * beta, 0])) > np.random.rand():
                                # accept the switch if try_L> cur_L or accept it according to boltzmann factor
                                best_time_inds = switch_inds
                                cur_L = try_L
                                n_switches += 1

                        iter += 1
                        if (iter % (500 * K)) == 0:
                            beta += 0.3 * beta + 0.03
                            std_factor /= 1.5
                            # print("Iteration {}; {} switches".format(iter, n_switches))

                    # if log2c > 0:
                    #     iterable_arr = np.asarray(list(itertools.product(np.asarray([-1, 0, 1]), repeat=K)))
                    #     # for n_fine_graining in range(1, log2c + 1):
                    #     half_log2c = int(log2c / 2)
                    #     for n_fine_graining in np.arange(1, log2c + 1)[np.hstack(
                    #             [half_log2c, np.arange(0, half_log2c), np.arange(half_log2c + 1, log2c)]
                    #     )]:  # it turns out it is better not to start from the largest step. again because of the asymmetric likelihood profile
                    #         all_switch_inds_arr = best_time_inds + iterable_arr * 2 ** (log2c - n_fine_graining)
                    #
                    #         # Coarse grained times are separated by c at most.
                    #         # However we want them to fluctuate -c to the left or +c to the right, and not -c/2 or c/2
                    #         # (because of the form of the likelihood which increases fast then decays slow).
                    #         # This can lead however to collisions if two indices change by c in the opposite directions.
                    #         # so we just filter these kind of states out.
                    #         ind1 = np.prod(np.diff(all_switch_inds_arr, axis=1) >= BAYESIAN_MIN_STATE_DURATION_FRAMES,
                    #                        axis=1)
                    #         ind2 = np.prod(all_switch_inds_arr >= 0 + BAYESIAN_MIN_STATE_DURATION_FRAMES,
                    #                        axis=1, dtype=bool)
                    #         ind3 = np.prod(all_switch_inds_arr <= N - BAYESIAN_MIN_STATE_DURATION_FRAMES,
                    #                        axis=1, dtype=bool)
                    #         all_switch_inds_arr = all_switch_inds_arr[np.where(ind1 & (ind2 & ind3))]
                    #         if not isempty(all_switch_inds_arr):
                    #             L_arr = [calcLikelihood(switch_inds, init_state) for switch_inds in all_switch_inds_arr]
                    #             best_time_inds = all_switch_inds_arr[np.argmax(L_arr)]

                    new_switch_time_inds = best_time_inds
                config_arr_K[j] = new_switch_time_inds
                if not np.any(np.isnan(new_switch_time_inds)):
                    L_arr_K[j], est_params = labeledParticleLikelihood(trajData, new_switch_time_inds, init_state,
                                                                       # enforceModelParameters=enforceModelParameters,
                                                                       returnParameters=True)
                else:
                    L_arr_K[j] = -np.inf
                if logMessages: print("Best time indices: {}".format(new_switch_time_inds))
            L_mat[i] = L_arr_K
            config_mat[i] = config_arr_K

        if logMessages: print(L_mat)
        if logMessages: print(config_mat)
        best_init_state = np.argmax([np.max(L_mat[0]), np.max(L_mat[1])])
        best_config = [best_init_state, config_mat[best_init_state][np.argmax(L_mat[best_init_state])]]
        best_L = np.max(L_mat[best_init_state])

    return [best_L, best_config]


def kEventsProb(K, init_state, T, T_stick, T_unstick):
    # probability of k events when distributed exponentially by tau1 tau2 etc...
    # iv is I_nu modified bessel function of first kind

    deltaT = np.abs(T_unstick - T_stick) / 2
    meanT = (T_stick + T_unstick) / 2

    if deltaT / meanT < 1e-6:  # do poisson if tau1~tau2
        P = (T / meanT) ** K / factorial(K) * np.exp(-T / meanT)

    else:
        # huge gaps between T stick and unstick lead to underflows, so make really short times the same as kind of short times
        if ((T_stick / T_unstick) > 100) | ((T_unstick / T_stick) > 100):
            if T_stick > T_unstick:
                T_unstick *= np.sqrt(T_stick / T_unstick)
            else:
                T_stick *= np.sqrt(T_unstick / T_stick)

        if init_state == 0:
            tau1 = T_stick
            tau2 = T_unstick
        elif init_state == 1:
            tau1 = T_unstick
            tau2 = T_stick
        else:
            print("Error! Invalid init state. Selecting 0 as default")
            tau1 = T_stick
            tau2 = T_unstick

        meanR = (1. / tau1 + 1. / tau2) / 2.
        deltaR = (1. / tau1 - 1. / tau2) / 2.
        sign_deltaR = np.sign(deltaR)
        deltaR = np.abs(deltaR)

        # do this for casting either by K or T
        T_new = T + 0. * K
        K_new = K + 0. * T

        T = T_new
        K = K_new

        zeroP = np.exp(-T / tau1)

        logPrefactor = -meanR * T + (K / 2) * np.log(T / (2 * deltaT)) - gammaln(np.floor(K / 2) + 1) \
                       + np.log(np.pi) / 2  # log to avoid overflow

        oddFactor = np.sqrt(tau2 / tau1) * iv(K / 2, deltaR * T)
        evenFactor = np.sqrt(deltaR * T / 2) * (iv((K - 1) / 2, deltaR * T) - sign_deltaR * iv((K + 1) / 2, deltaR * T))

        oddP = np.exp(logPrefactor) * oddFactor
        evenP = np.exp(logPrefactor) * evenFactor

        zeroInd = (K == 0)
        oddInd = K % 2 == 1
        evenInd = (K % 2 == 0) * ~zeroInd  # nonzero though

        if np.isscalar(K):
            if zeroInd:
                P = zeroP
            elif oddInd:
                P = oddP
            elif evenInd:
                P = evenP
        else:
            P = np.zeros(K.shape)
            P[zeroInd] = zeroP[zeroInd]
            P[oddInd] = oddP[oddInd]
            P[evenInd] = evenP[evenInd]
    return P


def extract_trajData_from_df(df):
    "Utility function to convert all the heavy data in the dataframe to an array of only the relevant data ('Trajdata')"
    t = df.t.values[:]
    if np.isnan(df.dt.values[0]):
        dt = df.dt.values  # [1:]  # delta t per step (first value is nan)
        dr2 = df.dr2.values  # [1:]  # delta r squared per step
    else:
        dt = df.dt.values  # delta t per step (first value is nan)
        dr2 = df.dr2.values  # delta r squared per step
    x = df.x.values
    y = df.y.values
    trajData = [t, dt, dr2, x, y]  # this eliminates the need for df
    return trajData

# def experimentLikelihood(df, T_stick, T_unstick, D, A, Parallelized=False, equal_particle_weight=True,
#                          calculationLog=False):
#     if calculationLog:
#         t_start = time.time()
#
#     particles_array = df.particle.unique()
#     grouped = df.groupby("particle")
#     particle_df = [grouped.get_group(i) for i in particles_array]
#
#     if equal_particle_weight:
#         weights = np.double(grouped.apply(len).values) ** (-1)  # inverse traj length
#     else:
#         weights = np.ones(len(particles_array))
#     weights = weights / sum(weights)  # normalize
#
#     if Parallelized:
#         args_iterable = []
#         logp_experiment = 0
#         with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
#             for j, logp_particles in enumerate(executor.map(labeledParticleLikelihood, particle_df, repeat(T_stick),
#                                                             repeat(T_unstick), repeat(D), repeat(A))):
#                 logp_experiment += logp_particles * weights[j]
#                 # print(j)
#         # for i in particles_array:
#         #     args_iterable.append(np.asarray((grouped.get_group(i), T_stick, T_unstick, D, A),dtype=object))
#         # # with Pool() as pool:
#         # with Pool(1) as pool:
#         #     logp_particles = pool.starmap(particleLikelihood, args_iterable)
#     else:
#         logp_particles = np.zeros(len(particles_array), dtype=object)
#         for i, particle_id in enumerate(particles_array):
#             logp_particles[i] = labeledParticleLikelihood(grouped.get_group(particle_id), T_stick, T_unstick, D, A)
#         logp_experiment = np.sum(logp_particles * weights)
#
#     if calculationLog:
#         try:
#             print("Estimated experiment likelihood of " + "{:.2f}".format(logp_experiment) + " in " + "{:.2f}".format(
#                 time.time() - t_start) + " sec")
#         except:
#             print("Estimated experiment likelihood derivatives in " + "{:.2f}".format(
#                 time.time() - t_start) + " sec")
#
#     if np.isnan(logp_experiment):
#         print("logp_experiment is nan - printing parameter values:")
#         print("T_st = ", T_stick, "\nT_un = ", T_unstick, "\nD = ", D, "\nA = ", A)
#
#     return logp_experiment


# def basicMaxLikelihood(df, x0, cons_arr=[0.1, 0.1, 1., 0.0001], scaling_factor=[100., 100., 10., 1000.], usePar=False,
#                        maxIter=100, savefile_extra="", fixedArguments=[False, False, False, False]):
#     # T_stick | T_unstick | D | A
#     # scaling factor > 1 is for increasing constraints and decreasing value in function (too small vars that make the constraint too small and the optimizer crosses it)
#     cons_arr = np.array(cons_arr)
#     scaling_factor = np.array(scaling_factor)
#
#     x0_orig = x0
#     x0 = x0 * scaling_factor
#     cons_arr = cons_arr * scaling_factor
#
#     # fixedArgs = np.where(np.asarray(fixedArguments))
#     variableArgs = np.where(~np.asarray(fixedArguments))[0]
#
#     L_all_args = lambda x: -1 * experimentLikelihood(df, x[0] / scaling_factor[0], x[1] / scaling_factor[1],
#                                                      x[2] / scaling_factor[2], x[3] / scaling_factor[3],
#                                                      Parallelized=usePar)
#
#     def conv_y2x(y):
#         y_arr = np.empty(len(x0), dtype=object)
#         count = 0
#         for i in range(len(x0)):
#             if fixedArguments[i]:
#                 y_arr[i] = x0[i]
#             else:
#                 y_arr[i] = y[count]
#                 count += 1
#         return y_arr
#
#     L = lambda y: L_all_args(conv_y2x(y))
#
#     cons_list = []
#     for j in range(len(variableArgs)):
#         cons_list.append({'type': 'ineq',
#                           'fun': lambda y: y[j] - cons_arr[variableArgs[j]]})
#     cons = tuple(cons_list)
#     # cons = ({'type': 'ineq',
#     #          'fun': lambda x: x[0] - cons_arr[0]},
#     #         {'type': 'ineq',
#     #          'fun': lambda x: x[1] - cons_arr[1]},
#     #         {'type': 'ineq',
#     #          'fun': lambda x: x[2] - cons_arr[2]},
#     #         {'type': 'ineq',
#     #          'fun': lambda x: x[3] - cons_arr[3]})
#
#     # autograd tensorflow...jax tfprobability
#     print("Beginning Minimization:")
#     t_start = time.time()
#     # minimum = scipy.optimize.minimize(L, x0, constraints=cons, options={'disp': True, 'maxiter': maxIter})
#     if usePar:
#         minimum = scipy.optimize.minimize(L, x0=x0[variableArgs], constraints=cons)  # , maxIter=maxIter)
#     else:
#         minimum = scipy.optimize.minimize(L, x0=x0[variableArgs], constraints=cons,
#                                           )  # jac=jacobian(L)),  # maxIter=maxIter)  # ,  hess=hessian(L))
#
#     filename = "likelihood_data_opt" + savefile_extra + ".npy"
#     filepath = os.path.join(OUTPUTS_DIR, filename)
#     backupBeforeSave(filepath)
#     warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#     minimum.x /= scaling_factor[variableArgs]
#     results_min = x0_orig
#     for j in range(len(variableArgs)):
#         results_min[variableArgs[j]] = minimum.x[j]
#
#     np.save(filepath, [results_min, minimum, x0_orig, df])
#     print("Saved results in " + filepath)
#     print("Total optimization time of " + "{:.2f}".format(time.time() - t_start) + " sec")
#     print(minimum)
#     print(results_min)
#     return results_min
#
#
# def gridLikelihood(df, T_stick_arr, T_unstick_arr, D_arr, A_arr, usePar=False, saveFiles=True):
#     args_iterable = itertools.product(T_stick_arr, T_unstick_arr, D_arr, A_arr)
#     args_list = list(args_iterable)
#     args_iterable = itertools.product([df], T_stick_arr, T_unstick_arr, D_arr, A_arr)
#     # parallelized = False  # parallelization is in the experiment likelihood
#     # if parallelized:
#     #     with Pool() as pool:
#     #         L_tensor = pool.starmap(experimentLikelihood, args_iterable)
#     # else:
#     t_start = time.time()
#
#     list_args_iterable = list(args_iterable)
#     L_tensor = np.zeros(len(list_args_iterable))
#     for i, args in enumerate(list_args_iterable):
#         L_tensor[i] = experimentLikelihood(args[0], args[1], args[2], args[3], args[4], Parallelized=usePar)
#     L_tensor = list(L_tensor)
#
#     print("Elapsed time = " + "{:.2f}".format(time.time() - t_start) + "sec")
#
#     # save here
#     if saveFiles:
#         filename = "likelihood_data_grid"
#         filepath = os.path.join(OUTPUTS_DIR, filename + ".npy")
#         backupBeforeSave(filepath)
#         warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#         np.save(filepath, [args_list, L_tensor, df])
#
#     max_likelihood_params = args_list[np.argmax(L_tensor)]
#     T_stick = max_likelihood_params[0]
#     T_unstick = max_likelihood_params[1]
#     D = max_likelihood_params[2]
#     A = max_likelihood_params[3]
#
#     L_sorted = np.sort(L_tensor)
#     print("Next highest log-Likelihoods: " + str(L_sorted[-4:-1] - L_sorted[-1]))
#     print("T_st = ", T_stick, "\nT_un = ", T_unstick, "\nD = ", D, "\nA = ", A)
#     return L_tensor
