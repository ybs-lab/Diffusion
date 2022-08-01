import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import BallTree
from extras import arr2string, string2arr, vec_translate, isempty
import scipy.spatial
from itertools import repeat, groupby
from bayesianTools import labelAllTrajectories

from config import DEFAULT_NEIGHBOR_THRESHOLD_UM, BREAK_COLLIDING_TRAJECTORIES_WIN_SIZE_FRAMES, \
    MAX_ALLOWED_COLLISIONS_IN_WIN, MIN_BROKEN_DF_LENGTH, MAX_PROCESSORS_LIMIT, BAYESIAN_MIN_STATE_DURATION_FRAMES, \
    MAX_FRAME_BEFORE_UNDERSAMPLING


def getDiffs(df, dt, t_column_name="frame", is_df_padded=True, needToGroupBy=True):
    if is_df_padded:  # if ALL frames are filled
        diffs_df = df[["x", "y", "particle", "trajDuration", "isNotPad"]]
        keep_inds = diffs_df.index[diffs_df["isNotPad"].values]
        if needToGroupBy:
            dxdy_df = diffs_df.groupby("particle").diff(int(dt))
        else:
            dxdy_df = diffs_df.diff(int(dt))
        dx = dxdy_df.loc[keep_inds, "x"].values
        dy = dxdy_df.loc[keep_inds, "y"].values
        particle = diffs_df.loc[keep_inds, "particle"].values
        trajDuration = diffs_df.loc[keep_inds, "trajDuration"].values

    else:
        particles = df.particle.unique()
        gb = df.groupby("particle")
        dx = []
        dy = []
        particle = []
        trajDuration = []
        for particle_id in particles:
            df_particle = gb.get_group(particle_id)
            t = df_particle[t_column_name].values
            x = df_particle.x.values
            y = df_particle.y.values
            curTrajDuration = len(df_particle)
            inds = np.vstack(
                np.where(np.triu(
                    scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(t[:, None], 'cityblock'))) == dt))
            particle = np.hstack([particle, particle_id * np.ones(len(inds[0, :]), dtype=int)])
            trajDuration = np.hstack([trajDuration, curTrajDuration * np.ones(len(inds[0, :]), dtype=int)])
            dx = np.hstack([dx, (x[inds[1, :]] - x[inds[0, :]])])
            dy = np.hstack([dy, (y[inds[1, :]] - y[inds[0, :]])])

    return [dx, dy, particle, trajDuration]


def intersecting_neighbors(df, isPar=False, threshold=DEFAULT_NEIGHBOR_THRESHOLD_UM):
    # 5um looks like a good number for experiments of March 2022
    default_string_no_neighbors = "None"

    df = df.copy()
    df["intersection_threshold"] = threshold  # for easy life just pass it with the df
    gb_exp = df.groupby(["experiment", "filename"])
    N_groups = gb_exp.ngroups
    all_experiments_df = [exp_df for _, exp_df in gb_exp]
    neighbors_array_per_experiment = np.empty(N_groups, dtype=object)
    N_neighbors_array_per_experiment = np.empty(N_groups, dtype=object)
    indices_per_experiment = np.empty(N_groups, dtype=object)
    print("Searching for colliding particles...")
    # Parallelization per experiment causes memory overflow - instead parallelizing per frame in findNeighbors_experiment.
    for j in range(N_groups):
        output = findNeighbors_experiment(all_experiments_df[j], isPar=isPar)
        neighbors_array_per_experiment[j] = output[0]
        N_neighbors_array_per_experiment[j] = output[1]
        indices_per_experiment[j] = output[2]
        # print("Found neighbors for file number " + str(j))

    indices = np.hstack(indices_per_experiment)
    N_neighbors = np.hstack(N_neighbors_array_per_experiment)
    neighbors = np.hstack(neighbors_array_per_experiment)
    df.loc[indices, "n_neighbors"] = N_neighbors
    df.loc[indices, "colliding_neighbors"] = neighbors  # this is a string instead of array
    df.colliding_neighbors = df.colliding_neighbors.replace(r'^\s*$', default_string_no_neighbors, regex=True)

    if np.isin("intersection_threshold", df.columns):
        df = df.drop(columns=["intersection_threshold"])

    return df


def findNeighbors_experiment(df, isPar=False):
    gb_t = df.groupby("video_frame")
    Nframes = gb_t.ngroups
    all_frames_df = [t_df for _, t_df in gb_t]
    neighbors_array_per_frame = np.empty(Nframes, dtype=object)
    df_indices_array_per_frame = np.empty(Nframes, dtype=object)
    N_neighbors_array_per_frame = np.empty(Nframes, dtype=object)

    if (len(df.particle.unique()) > 1) & (df.experiment.unique()[0].lower() != "synthetic"):
        if isPar:
            with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
                for k, output in enumerate(executor.map(findNeighbors_df, all_frames_df)):
                    neighbors_array_per_frame[k] = output[0]
                    N_neighbors_array_per_frame[k] = output[1]
                    df_indices_array_per_frame[k] = output[2]
                    # print("Found neighbors for frame number " + str(k) + " (Parallel)")
        else:
            for k in range(Nframes):
                output = findNeighbors_df(all_frames_df[k])
                neighbors_array_per_frame[k] = output[0]
                N_neighbors_array_per_frame[k] = output[1]
                df_indices_array_per_frame[k] = output[2]
                # print("Found neighbors for frame number " + str(k))

        neighbors_arr = np.hstack(neighbors_array_per_frame)
        N_neighbors_arr = np.hstack(N_neighbors_array_per_frame)
        indices_arr = np.hstack(df_indices_array_per_frame)
    else:  # only 1 particle so no neighbors
        neighbors_arr = np.empty(len(df), dtype=object)
        neighbors_arr[:] = ''
        N_neighbors_arr = np.zeros(len(df))
        indices_arr = np.asarray(df.index)
    return [neighbors_arr, N_neighbors_arr, indices_arr]


def findNeighbors_df(frame_df):
    threshold = frame_df["intersection_threshold"].values[0]
    X = frame_df[["x", "y"]].values  # Nx2 matrix of points
    P = frame_df["particle"].values.astype(int)  # Nx1 matrix of particle ID

    # need to handle nan because it does problems in findNeighbors
    nan_ind = np.isnan(X[:, 0])
    X[nan_ind, :] = 100000.  # stam default value that doesnt collide with others

    neighbors_ind = findNeighbors(X, threshold)

    neighbors = np.asarray(
        [arr2string(P[np.setdiff1d(ind, np.union1d(j, nan_ind))], delimiter=',') for j, ind in
         enumerate(neighbors_ind)],
        dtype=object)  # remove self neighbor
    N_neighbors = np.asarray(
        [len(P[np.setdiff1d(ind, np.union1d(j, nan_ind))]) for j, ind in enumerate(neighbors_ind)])
    df_indices = np.asarray(frame_df.index)

    if any(nan_ind):
        neighbors[nan_ind] = ''
        N_neighbors[nan_ind] = 0
    return [neighbors, N_neighbors, df_indices]


def findNeighbors(X, threshold):
    tree = BallTree(X)
    neighbors_ind = tree.query_radius(X, threshold, return_distance=False, count_only=False, sort_results=False)
    return neighbors_ind


def calc_rolling_R_gyration(df, isPar=False):
    df = df.copy()
    particles = df.particle.unique()
    all_particle_df = [particle_df for _, particle_df in df.groupby("particle")]
    Rg2_array_per_particle = [np.empty(0, dtype=float) for i in particles]

    print("Calculating radius of gyration...")
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for j, output in enumerate(executor.map(calc_Rg_df, all_particle_df)):
                Rg2_array_per_particle[j] = output
                # print("Calculated Rg2 for particle " + str(j) + " (Parallel)")
    else:
        for j, particle in enumerate(particles):
            output = calc_Rg_df(all_particle_df[j])
            Rg2_array_per_particle[j] = output
            # print("Calculated Rg2 for particle " + str(j))

    df["Rg2_expanding"] = np.hstack(Rg2_array_per_particle)
    return df


def calc_Rg_df(df):
    N = len(df)
    X = df[["x", "y"]].values
    return calc_Rg_np(X, N)


def calc_Rg_np(X, N_elements):
    # Efficient calculation of radius of gyration: Rg^2 = 1/(2N^2) sum_ij (x_i-x_j)^2
    # Expand this to Rg^2_N+1 = [N^2 Rg^2_N +Nx^2_N+1+mom2_N -2X_N+1 * mom1_N]/(N+1)^2
    # where mom1_N = sum_1^N x_n ;  mom2_N = sum_1^N x_n^2
    # x is N x d mat. calculate radius of gyration as more and more data is added (the N rows)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    d = X.shape[1]
    Rg2_mat = np.zeros(X.shape)
    Rg2_mat[0, :] = np.nan  # nicer than zeros I think
    mom1_mat = np.nancumsum(X, 0)
    mom2_mat = np.nancumsum(X ** 2, 0)
    N_mat = np.cumsum(~np.isnan(X[:, 0]), 0,
                      dtype=float)  # how many not nan elements. we assume full rows are nan, not partial
    Rg2_prev = 0
    Rg2_new = np.empty([1, d])  # until two non-nan elements are met
    Rg2_new[:] = np.nan
    for n in range(N_elements - 1):
        x = X[n + 1, :]
        x_sq = x ** 2
        if np.any(~np.isnan(x)):
            mom1 = mom1_mat[n, :]
            mom2 = mom2_mat[n, :]
            N = N_mat[n]
            Rg2_new = (N ** 2 * Rg2_prev + N * x_sq + mom2 - 2 * x * mom1) / (N + 1) ** 2
            Rg2_prev = Rg2_new

        Rg2_mat[n + 1, :] = Rg2_new  # until two non-nan elements are encountered, this will be nan

    Rg2_arr = np.sum(Rg2_mat, 1)
    return Rg2_arr


def assignAllTrajStates(df, isPar=False):
    return labelAllTrajectories(df,isPar)


# def assignParticleTrajState(df_particle, T_stick, T_unstick, D, A, baseLen=64., skipImmobile=True):
#     # Segment the particle trajectory to smaller parts and assign states to them by maximum likelihood, given specific parameters
#     if len(df_particle) <= baseLen / 2:
#         df = df_particle.copy()
#         df["state"] = 0.
#         df["logL_states"] = 0.
#         logL_total = 0.
#         return [df.index, df.state.values, df.logL_states.values, logL_total]
#
#     if skipImmobile & ~(df_particle.isMobile.values[0]):  # skip immobile particles - they are just stuck
#         df = df_particle.copy()
#         df["state"] = 1.
#         df["logL_states"] = 0.
#         logL_total = 0.
#         return [df.index, df.state.values, df.logL_states.values, logL_total]
#
#     index_mat = segmentTraj(len(df_particle), baseLen=baseLen)
#     N_segments = index_mat.shape[0]
#     df = df_particle.copy()
#     df["state"] = 0.
#     df["logL_states"] = 0.
#     logL_total = 0.
#     for i in range(N_segments):
#         df_out, max_likelihood = particleLikelihood(df[index_mat[i, 0]:(index_mat[i, 1] + 1)], T_stick, T_unstick, D, A,
#                                                     1)
#         states_assigned = df_out["state"].values
#
#         # remove state=0.5 from the edges of middle segments
#         if i != 0:
#             states_assigned[0] = states_assigned[1]
#         if i != N_segments - 1:
#             states_assigned[-1] = states_assigned[-2]
#
#         df.loc[df.index[index_mat[i, 0]:(index_mat[i, 1] + 1)], "state"] = states_assigned
#         df.loc[df.index[index_mat[i, 0]:(index_mat[i, 1] + 1)], "logL_states"] = max_likelihood
#         logL_total += max_likelihood
#     # return df
#     return [df.index, df.state.values, df.logL_states.values, logL_total]
#
#
# def assignAllTrajStates(df_in, params_path, baseLen=64., isPar=False):
#     parameters_table = pd.read_csv(params_path)
#     table_experiments = parameters_table.experiment.unique()
#
#     df = df_in.copy()
#     all_experiments = df.experiment.unique()
#     N_experiments = len(all_experiments)
#     df["state"] = 0.
#     df["logL_states"] = 0.
#     gb_exp = df.groupby("experiment")
#
#     for exp_id, experiment in enumerate(all_experiments):
#         if not np.isin(experiment, table_experiments):
#             continue
#         cur_params = parameters_table[parameters_table.experiment == experiment]
#         # obviously not elegant
#         T_stick = cur_params.T_stick.values[0]
#         T_unstick = cur_params.T_unstick.values[0]
#         D = cur_params.D.values[0]
#         A = cur_params.A.values[0]
#         exp_df = gb_exp.get_group(experiment)
#         particles = exp_df.particle.unique()
#         N_particles = len(particles)
#         gb_particle = exp_df.groupby("particle")
#         all_particle_df = [gb_particle.get_group(i) for i in particles]
#         if isPar:
#             with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
#                 for j, output in enumerate(executor.map(assignParticleTrajState, all_particle_df, repeat(T_stick),
#                                                         repeat(T_unstick), repeat(D), repeat(A), repeat(baseLen))
#                                            ):
#                     df.loc[output[0], "state"] = output[1]
#                     df.loc[output[0], "logL_states"] = output[2]
#                     df.loc[output[0], "logL_total"] = output[3]
#                     print("Assigned states to particle " + str(j) + " of " + str(N_particles) + " ; Experiment " + str(
#                         exp_id) + "/" + str(N_experiments))
#         else:
#             for j, particle in enumerate(particles):
#                 output = assignParticleTrajState(all_particle_df[j], T_stick, T_unstick, D, A, baseLen=baseLen)
#                 df.loc[output[0], "state"] = output[1]
#                 df.loc[output[0], "logL_states"] = output[2]
#                 df.loc[output[0], "logL_total"] = output[3]
#                 print("Assigned states to particle " + str(j) + " of " + str(N_particles) + " ; Experiment " + str(
#                     exp_id) + "/" + str(N_experiments))
#
#         print("Assigned states to experiment " + experiment)
#
#     return df


def fit_D_n_particle(df_particle):
    trajDuration = df_particle.head(1).trajDuration.unique()[0]
    fps = df_particle.head(1).fps.unique()[0]
    dt_arr = np.arange(1, np.min([200, np.floor(trajDuration / 2)]), step=3)
    MSD_arr = 0 * dt_arr
    for k, dt in enumerate(dt_arr):
        out = getDiffs(df_particle, dt, is_df_padded=True, needToGroupBy=False)
        dr2 = out[0] ** 2 + out[1] ** 2
        if len(dr2) >= 3:
            MSD_arr[k] = np.nanmean(dr2)
        else:
            MSD_arr[k] = np.nan
    ind = (np.isnan(MSD_arr)) | (MSD_arr <= 0)
    dt_arr = dt_arr[~ind] / fps
    MSD_arr = MSD_arr[~ind]
    try:
        poly_coeffs = np.polyfit(np.log(dt_arr), np.log(MSD_arr), 1, w=dt_arr ** (-1))
        n_fit = poly_coeffs[0]
        D_fit = np.exp(poly_coeffs[1]) / 4
    except:
        n_fit = np.nan
        D_fit = np.nan
    return [n_fit, D_fit]


def assign_D_n_estimate(df, isPar=False):
    particles = df.particle.unique()
    gb = df.groupby("particle")
    all_particle_df = [gb.get_group(i) for i in particles]
    n_array = 0. * particles
    D_array = 0. * particles
    print("Assigning crude estimates of diffusion coefficients...")
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for j, output in enumerate(executor.map(fit_D_n_particle, all_particle_df)):
                n_array[j] = output[0]
                D_array[j] = output[1]
                # print("Estimated diffusion coefficients for particle " + str(j) + " (Parallel)")
    else:
        for j, particle in enumerate(particles):
            output = fit_D_n_particle(all_particle_df[j])
            n_array[j] = output[0]
            D_array[j] = output[1]
            # print("Estimated diffusion coefficients for particle " + str(j))

    n_dict = {p: n for p, n in zip(particles, n_array)}
    D_dict = {p: D for p, D in zip(particles, D_array)}
    df["n_est"] = vec_translate(df["particle"].values, n_dict)
    df["D_est"] = vec_translate(df["particle"].values, D_dict)
    return df


def df_calculate_local_D(particle_df):
    T = particle_df["T_window"].values[0]
    indices = particle_df.index
    D_local = (particle_df.rolling(window=T, center=True).x.var(skipna=True) + \
               particle_df.rolling(window=T, center=True).y.var(skipna=True)) \
              * particle_df.fps.values[0] / float(T)
    return [D_local.values, D_local.index]


def calculate_local_D(df, T=100, isPar=False):
    df = df.copy()
    df["T_window"] = T
    particles = df.particle.unique()
    local_D_array_per_particle = [np.empty(0, dtype=object) for i in particles]
    df_indices_array_per_particle = [np.empty(0, dtype=object) for i in particles]
    all_particle_df = [particle_df for _, particle_df in df.groupby("particle")]

    print("Calculating local diffusion coefficients...")
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for j, output in enumerate(executor.map(df_calculate_local_D, all_particle_df)):
                local_D_array_per_particle[j] = output[0]
                df_indices_array_per_particle[j] = output[1]
                # print("Estimated local diffusion coefficients for particle " + str(j) + " (Parallel)")
    else:
        for j, particle in enumerate(particles):
            output = df_calculate_local_D(all_particle_df[j])
            local_D_array_per_particle[j] = output[0]
            df_indices_array_per_particle[j] = output[1]
            # print("Estimated local diffusion coefficients for particle " + str(j))

    df.loc[np.hstack(df_indices_array_per_particle), "D_local_T_" + str(T)] = np.hstack(local_D_array_per_particle)

    if np.isin("T_window", df.columns):
        df = df.drop(columns=["T_window"])

    return df


def generateMeanDiffusionDf(df, factor=4.):
    gb = df.groupby("particle")
    # improve this!
    return pd.concat([factor * gb.D_est.first(),
                      factor * gb.D_local_T_30.mean(),
                      factor * gb.D_local_T_90.mean(),
                      factor * gb.D_local_T_150.mean(),
                      factor * 3. * (gb.Rg2_expanding.last() / gb.t.last().values).rename("Rg2/T"),
                      factor * ((gb.x.last() - gb.x.first()) ** 2 / gb.t.last().values + \
                                (gb.y.last() - gb.y.first()).values ** 2 / gb.t.last().values).rename("E2E") / 4.,
                      gb.t.last().rename("t_end"),
                      gb.experiment.first()
                      ], axis=1)


def shiftTrajToOrigin(df):
    df = df.copy()
    df['x'] = df['x'] - df.groupby('particle')['x'].transform('first')
    df['y'] = df['y'] - df.groupby('particle')['y'].transform('first')
    return df


def smooth_states(df, min_state_duration_frames=3 * BAYESIAN_MIN_STATE_DURATION_FRAMES, isPar=False):
    particle_df_array = [particle_df for _, particle_df in df.groupby("particle")]
    smoothed_states_array = np.empty(len(particle_df_array), dtype=object)
    print(
        "Smoothing particle states (temporary fix because of the problematic labeling!)")
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for i, output in enumerate(
                    executor.map(smooth_particle_df_states, particle_df_array, repeat(min_state_duration_frames))):
                smoothed_states = output
                smoothed_states_array[i] = smoothed_states
                # print("Smoothed states for particle {} (Parallel)".format(i))

    else:
        for i, particle_df in enumerate(particle_df_array):
            smoothed_states = smooth_particle_df_states(particle_df, min_state_duration_frames)
            smoothed_states_array[i] = smoothed_states
            # print("Smoothed states for particle {}".format(i))

    # df["smoothed_state"]=0. #to remove pesky warning
    df = df.copy()
    df["smoothed_state"] = np.hstack(smoothed_states_array)
    return df


def smooth_particle_df_states(particle_df, min_state_duration_frames):
    states = particle_df.state.values
    # print(states)
    states[0] = states[1]
    states[-1] = states[-2]
    states = np.floor(states)
    states[1 + np.where(np.abs(np.diff(states)) == 1)[0]] = 0.5
    states[0] = 0.5
    states[-1] = 0.5
    split_states = np.split(states, np.where(states == 0.5)[0])[1:-1]  # this is missing the last 0.5
    len_arr = [len(seg) for seg in split_states]
    state_arr = [seg[1] for seg in split_states]
    valid_segs_ind = np.where(np.asarray(len_arr) >= min_state_duration_frames)[0]
    if (len(split_states) <= 1) | (len(valid_segs_ind) == 0):
        smoothed_states = states
    else:
        smoothed_seg_arr = np.empty(len(split_states) + 1, dtype=object)
        flip_first_entry_flag = False
        for n, seg in enumerate(split_states):
            if len(seg) < min_state_duration_frames:
                if np.any([valid_segs_ind < n]):
                    prev_large_segment = np.max(valid_segs_ind[valid_segs_ind < n])
                else:
                    prev_large_segment = []
                if np.any([valid_segs_ind > n]):
                    next_large_segment = np.min(valid_segs_ind[valid_segs_ind > n])
                else:
                    next_large_segment = []
                if isempty(prev_large_segment):
                    chosen_seg = next_large_segment
                elif isempty(next_large_segment):
                    chosen_seg = prev_large_segment
                else:
                    if len_arr[prev_large_segment] >= len_arr[next_large_segment]:
                        chosen_seg = prev_large_segment
                    else:
                        chosen_seg = next_large_segment
                # print("Segment no. {} has length of {}, segment no. {} has length of {}".format
                #       (prev_large_segment, len_arr[prev_large_segment], next_large_segment,
                #        len_arr[next_large_segment]))
                # print("Converting it to value of segment no. {}".format(chosen_seg))

                smoothed_seg_arr[n] = 0 * seg + state_arr[chosen_seg]
                flip_first_entry_flag = True
            else:
                if flip_first_entry_flag:
                    seg[0] = seg[1]
                smoothed_seg_arr[n] = seg
                flip_first_entry_flag = False
        smoothed_seg_arr[-1] = [0.5]
        smoothed_states = np.hstack(smoothed_seg_arr)
    return smoothed_states


def breakTrajectories_Collisions(df, win_size=BREAK_COLLIDING_TRAJECTORIES_WIN_SIZE_FRAMES,
                                 max_allowed_collisions_in_window=MAX_ALLOWED_COLLISIONS_IN_WIN,
                                 min_df_len=MIN_BROKEN_DF_LENGTH, isPar=False):
    particle_df_array = [p_df for _, p_df in df.groupby("particle")]
    particle_df_partition_array = np.empty(len(particle_df_array), dtype=object)

    print("Breaking trajectories due to collisions...")
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for i, output in enumerate(executor.map(segmentate_particle_df, particle_df_array, repeat(win_size),
                                                    repeat(max_allowed_collisions_in_window), repeat(min_df_len))):
                particle_df_partition_array[i] = output[0]
                # print("Collision handling: split particle {} trajectory to {} segments, kept {}% of data (Parallel)"
                #       .format(i, len(output[0]), np.round(100. * output[1], 2)))
    else:
        for i, particle_df in enumerate(particle_df_array):
            output = segmentate_particle_df(particle_df, win_size, max_allowed_collisions_in_window, min_df_len)
            particle_df_partition_array[i] = output[0]
            # print("Collision handling: split particle {} trajectory to {} segments, kept {}% of data"
            #       .format(i, len(output[0]), np.round(100. * output[1], 2)))

    return particle_df_partition_array


def segmentate_particle_df(particle_df, win_size, max_allowed_collisions_in_window, min_df_len):
    N = len(particle_df)
    collisions_in_window = (particle_df["n_neighbors"] > 0).astype(int).rolling(window=win_size, center=True,
                                                                                min_periods=1).sum().values
    mask_to_remove = collisions_in_window > max_allowed_collisions_in_window
    if isempty(mask_to_remove):
        split_df_arr = [particle_df]
        output = [split_df_arr, 1.]
    else:
        indices = np.nonzero(mask_to_remove[1:] != mask_to_remove[:-1])[0] + 1
        segs_to_keep = np.split(np.arange(N), indices)
        segs = segs_to_keep[0::2] if ~mask_to_remove[0] else segs_to_keep[1::2]
        len_arr = np.asarray([len(seg) for seg in segs])
        segs = np.asarray(segs, dtype=object)[len_arr >= min_df_len]
        if len(segs) == 0:
            split_df_arr = []
            output = [split_df_arr, 0.]
        else:
            df_idx = particle_df.index.to_numpy().astype(int)
            split_df_arr = [particle_df.loc[df_idx[np.asarray(seg, dtype=int)]] for seg in segs]
            output = [split_df_arr, sum(len_arr[len_arr >= min_df_len]) / N]
    return output


def undersampleAllParticles(df, max_allowed_frame=MAX_FRAME_BEFORE_UNDERSAMPLING, isPar=False):
    particle_df_array = [particle_df for _, particle_df in df.groupby("particle")]
    factor_arr = np.asarray([np.ceil(len(particle_df) / max_allowed_frame) for particle_df in particle_df_array],dtype=int)
    output_particle_df_array = np.empty(len(particle_df_array), dtype=object)
    print("Undersampling long trajectories (max allowed frames = {})".format(max_allowed_frame))
    if isPar:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for k, output in enumerate(executor.map(undersampleTraj, particle_df_array, factor_arr)):
                output_particle_df_array[k] = output
    else:
        for i, particle_df in enumerate(particle_df_array):
            output_particle_df_array[i] = undersampleTraj(particle_df, factor_arr[i])

    df = pd.concat(output_particle_df_array).reset_index(drop=True)
    return df


def undersampleTraj(particle_df, factor):
    if factor > 1:
        df = particle_df[::factor].copy() #copy to deal with the pesky warning!
        df.fps /= factor
        frames = df.frame.values
        init_frame  = frames[0]
        df.frame = (init_frame + (frames - init_frame) / factor).astype(int)
        # df = addDisplacement(df, "x")
        # df = addDisplacement(df, "y")
        # df = addDisplacement(df, "t")  # must for bayesianTools
        # df = addTrajDuration(df)
    else:
        df = particle_df
    return df
