import os
import numpy as np
import pandas as pd
import scipy.spatial
import glob
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import extras
from bayesianTools import segmentTraj, particleLikelihood

TRAJECTORIES_DIR = ".\\Trajectories"
DATA_DIR = ".\\Data"
IMAGING_PARAMETERS_PATH = ".\\Resources\\Parameters\\imaging_parameters_table.csv"
MODEL_PARAMETERS_PATH = ".\\Resources\\Parameters\\model_parameters_table.csv"


def renameParticleID(df, offset):
    # Rename particle IDs because different files might have the same particle ID (but the particles are different)
    # This alters the input dataframe!
    particleIDs = pd.unique(df.particle)
    Nparticles = len(particleIDs)
    dict_ID_to_index = dict(zip(particleIDs, offset + np.arange(Nparticles)))
    df.particle = vec_translate(df.particle, dict_ID_to_index)
    return Nparticles


def readFromCSV(traj_path, threshold=1000000):
    # Return a pandas dataframe with all the experimental data found in input path.
    # In directory traj_path, each different folder is considered a different experiment
    # threshold is related to breaking of long trajectories, not recommended to use!
    #
    # Notes:
    # Multiple files are concatenated and particle IDs are renamed such that all particle IDs are unique
    # Also drop all columns except permitted ones

    allowed_columns = ['particle', 'frame', 'x', 'y', 'mass']  # in the desired order
    offset = 0
    experiments = os.listdir(traj_path)
    if len(experiments) == 0:
        print("No folders found in directory: " + traj_path)
        df = []
        return

    for n_exp, experiment in enumerate(experiments):
        fullpath = traj_path + "\\" + experiment
        files = os.listdir(fullpath)
        if len(files) == 0:
            print("No files found in directory: " + fullpath)
            continue
        else:
            for n, file in enumerate(files):
                cur_df = pd.read_csv(fullpath + "\\" + file)
                cur_df.drop(cur_df.columns.difference(allowed_columns), axis=1,
                            inplace=True)  # remove unwanted columns
                cur_df = cur_df[allowed_columns]  # reorder columns
                cur_df["particle"] = cur_df["particle"].astype('int')  # convert to int
                cur_df["experiment"] = experiment
                # make sure the dataframe is well sorted
                cur_df = cur_df.sort_values(["particle", "frame"])
                cur_df.index = range(len(cur_df))
                if threshold <= 1000:
                    cur_df = breakLongTrajectory(cur_df, threshold)
                N_particles = renameParticleID(cur_df, offset)
                offset = offset + N_particles
                if "df" not in locals():
                    df = cur_df
                else:
                    df = df.append(cur_df, "ignore_index")  # add new df on top of existing df
    return df


def breakLongTrajectory(df, threshold):
    particles = df.particle.unique()
    columns_index = np.where(df.columns == "particle")[0][0]
    N_particles = -1
    for j, particle_id in enumerate(particles):
        df_particle = df[df["particle"] == particle_id]
        if len(df_particle) > 2 * threshold:
            break_indices = np.arange(len(df_particle), step=threshold)
            break_indices[-1] = len(df_particle)
            break_indices = break_indices + df_particle.index[0]
            for i in range(len(break_indices) - 1):
                df_particle.at[break_indices[i]:break_indices[i + 1], "particle"] = N_particles + 1
                N_particles = N_particles + 1
        else:
            df_particle.at[:, "particle"] = N_particles + 1
            N_particles = N_particles + 1
        if j == 0:
            df_out = df_particle
        else:
            df_out = df_out.append(df_particle, "ignore_index")

    return df_out


def setInitFrame(df):
    # Make sure the initial frame of each particle is 0
    df['frame'] = df['frame'] - df.groupby('particle')['frame'].transform('first')
    return df


def addDisplacement(df, ax):
    # ax = "x" or "y" or "t"
    # Add time column
    columns = df.columns
    diff_ax_name = "d" + ax
    if len(np.argwhere(columns == diff_ax_name)) == 0:  # no displacement column yet
        copy_df = df[["particle", ax]]
        df[diff_ax_name] = copy_df.groupby("particle").diff()  # this makes sure there is NaN at interparticle rows
        # reorder columns
        ax_ind = int(np.argwhere(columns == ax)) + 1
        columns_new = np.append(columns[np.arange(ax_ind)], diff_ax_name)
        columns_new = np.append(columns_new, columns[ax_ind + np.arange(len(columns) - ax_ind)])
        df = df[columns_new]
    return df


def removeImmobileParticle(df, imaging_params_path):
    # t_record in sec, cutoff_dist in micrometers
    try:
        parameters_table = pd.read_csv(imaging_params_path)
    except:
        return df

    allowed_experiments = parameters_table[parameters_table.Remove_Immobile].experiment.values
    t_record_list = parameters_table[parameters_table.Remove_Immobile].Remove_Immobile_t_record.values
    cutoff_dist_list = parameters_table[parameters_table.Remove_Immobile].Remove_Immobile_cutoff_dist.values

    D = df.copy()[["experiment", "particle", "t", "x", "y"]]
    D.loc[:, "keepIndex"] = True
    particles = D.particle.unique()
    # this is not really msd but just (r(t)-r(0))^2
    # D['MSD'] = (D['x'] - D.groupby('particle')['x'].transform('first')).pow(2) + (
    #         D['y'] - D.groupby('particle')['y'].transform('first')).pow(2)
    gb_exp = df.groupby("experiment")
    for n_iter, experiment in enumerate(allowed_experiments):
        df_exp = gb_exp.get_group(experiment)
        particles = df_exp.particle.unique()
        G = df_exp.groupby("particle")
        for particle_id in particles:
            particle_df = G.get_group(particle_id)
            t_record = t_record_list[n_iter]
            cutoff_dist = cutoff_dist_list[n_iter]

            particle_df_trimmed = particle_df[particle_df["t"] <= t_record]
            x = particle_df_trimmed.x.values
            y = particle_df_trimmed.y.values
            dx = scipy.spatial.distance.pdist(x[:, None], 'cityblock')
            dy = scipy.spatial.distance.pdist(y[:, None], 'cityblock')
            if np.max(dx) < cutoff_dist and np.max(dy) < cutoff_dist:
                D.loc[particle_df.index, "keepIndex"] = False

    return df[D["keepIndex"]].reset_index(drop=True)


def addTrajDuration(df):  # in frames
    df = df.copy()
    df["trajDuration"] = 0.
    df["trajDurationSec"] = 0.
    len_array = df.groupby("particle").apply(len)
    len_dict = len_array.to_dict()
    # groupby experiment to avoid memory issues
    experiments = df["experiment"].unique()
    gb_exp = df.groupby("experiment")
    for n, experiment in enumerate(experiments):
        df_exp = gb_exp.get_group(experiment)
        len_df_exp = len(df_exp)
        indices = df_exp.index
        particles = df_exp["particle"].values
        fps = df_exp.head(1).fps.values[0]
        trajDuration = vec_translate(particles, len_dict)  # use basic vectorized dictionary function
        trajDurationSec = trajDuration / fps
        df.loc[df_exp.index, "trajDuration"] = trajDuration
        df.loc[df_exp.index, "trajDurationSec"] = trajDurationSec

    return df


def assignParticleTrajState(df_particle, T_stick, T_unstick, D, A, baseLen=64.):
    # Segment the particle trajectory to smaller parts and assign states to them by maximum likelihood, given specific parameters
    index_mat = segmentTraj(len(df_particle), baseLen=baseLen)
    N_segments = index_mat.shape[0]
    df = df_particle.copy()
    df["state"] = 0.
    df["logL_states"] = 0.
    for i in range(N_segments):
        df_out, max_likelihood = particleLikelihood(df[index_mat[i, 0]:(index_mat[i, 1] + 1)], T_stick, T_unstick, D, A,
                                                    1)
        states_assigned = df_out["state"].values

        # remove state=0.5 from the edges of middle segments
        if i != 0:
            states_assigned[0] = states_assigned[1]
        if i != N_segments - 1:
            states_assigned[-1] = states_assigned[-2]

        df.loc[df.index[index_mat[i, 0]:(index_mat[i, 1] + 1)], "state"] = states_assigned
        df.loc[df.index[index_mat[i, 0]:(index_mat[i, 1] + 1)], "logL_states"] = max_likelihood

    # return df
    return [df.index, df.state.values, df.logL_states.values]


def assignAllTrajStates(df_in, params_path, baseLen=64., isPar=False):
    parameters_table = pd.read_csv(params_path)

    df = df_in.copy()
    all_experiments = df.experiment.unique()
    N_experiments = len(all_experiments)
    df["state"] = 0.
    df["logL_states"] = 0.
    gb_exp = df.groupby("experiment")

    for exp_id, experiment in enumerate(all_experiments):
        cur_params = parameters_table[parameters_table.experiment == experiment]
        # obviously not elegant
        T_stick = cur_params.T_stick.values[0]
        T_unstick = cur_params.T_unstick.values[0]
        D = cur_params.D.values[0]
        A = cur_params.A.values[0]
        exp_df = gb_exp.get_group(experiment)
        particles = exp_df.particle.unique()
        N_particles = len(particles)
        gb_particle = exp_df.groupby("particle")
        all_particle_df = [gb_particle.get_group(i) for i in particles]
        if isPar:
            with ProcessPoolExecutor() as executor:
                for j, output in enumerate(executor.map(assignParticleTrajState, all_particle_df, repeat(T_stick),
                                                        repeat(T_unstick), repeat(D), repeat(A), repeat(baseLen))
                                           ):
                    df.loc[output[0], "state"] = output[1]
                    df.loc[output[0], "logL_states"] = output[2]
                    print("Assigned states to particle " + str(j) + " of " + str(N_particles) + " ; Experiment " + str(
                        exp_id) + "/" + str(N_experiments))
        else:
            for j, particle in enumerate(particles):
                output = assignParticleTrajState(all_particle_df[j], T_stick, T_unstick, D, A, baseLen=baseLen)
                df.loc[output[0], "state"] = output[1]
                df.loc[output[0], "logL_states"] = output[2]
                print("Assigned states to particle " + str(j) + " of " + str(N_particles) + " ; Experiment " + str(
                    exp_id) + "/" + str(N_experiments))

        print("Assigned states to experiment " + experiment)

    return df


def get_missingFrames(particle_df):
    frames = particle_df.frame.values
    min_frame = np.min(frames)
    max_frame = np.max(frames)
    missingFrames = np.where(~np.isin(np.arange(min_frame, max_frame), frames))[0]
    N_missing_frames = len(missingFrames)
    missingFrames_df = pd.DataFrame([])
    if N_missing_frames > 0:
        base_df = particle_df.head(1).copy()
        base_df["x"] = np.nan
        base_df["y"] = np.nan
        base_df["t"] = np.nan
        base_df["mass"] = np.nan
        base_df["isNotPad"] = False
        missingFrames_df = base_df.loc[base_df.index.repeat(N_missing_frames)]
        missingFrames_df.index = np.arange(N_missing_frames)
        missingFrames_df.loc[:, "frame"] = missingFrames

    return missingFrames_df


def addPadding(df, isPar=False):
    df = df.copy()
    df["isNotPad"] = True
    particles = df.particle.unique()
    gb = df.groupby("particle")
    all_particle_df = [gb.get_group(i) for i in particles]
    missingFrame_df_array = [pd.DataFrame([]) for i in particles]

    if isPar:
        with ProcessPoolExecutor() as executor:
            for j, output in enumerate(executor.map(get_missingFrames, all_particle_df)):
                missingFrame_df_array[j] = output
                #print("Padded particle " + str(j))
    else:
        for j, particle in enumerate(particles):
            output = get_missingFrames(all_particle_df[j])
            missingFrame_df_array[j] = output
            #print("Padded particle " + str(j))

    df_to_append = pd.concat(missingFrame_df_array, ignore_index=True)
    df_to_append.index = df_to_append.index + df.index.max() + 1
    df = df.append(df_to_append)
    df = df.sort_values(["particle", "frame"])
    df = df.reset_index(drop=True)
    return df


# def addPadding(df):
#     df = df.copy()
#     df["isNotPad"] = True
#     particles = df.particle.unique()
#     gb = df.groupby("particle")
#     max_index = df.index.max() + 1
#     offset = 0
#     for particle in particles:
#         particle_df = gb.get_group(particle)
#         frames = particle_df.frame.values
#         min_frame = np.min(frames)
#         max_frame = np.max(frames)
#         missingFrames = np.where(~np.isin(np.arange(min_frame, max_frame), frames))[0]
#         N_missing_frames = len(missingFrames)
#         if N_missing_frames > 0:
#             base_df = particle_df.head(1).copy()
#             base_df["x"] = np.nan
#             base_df["y"] = np.nan
#             base_df["t"] = np.nan
#             base_df["mass"] = np.nan
#             base_df["isNotPad"] = False
#             df_to_add = base_df.loc[base_df.index.repeat(N_missing_frames)]
#             df_to_add.index = max_index + offset + np.arange(N_missing_frames)
#             df_to_add.loc[:, "frame"] = missingFrames
#             offset = offset + N_missing_frames
#             df = df.append(df_to_add)
#
#     df = df.sort_values(["particle", "frame"])
#     df = df.reset_index(drop=True)
#     return df


def importAll(getLatest=True, threshold=1000000, assignTrajStates=False, isPar=False, returnPadded=True):
    df_list = glob.glob(DATA_DIR + "\\df*.fea")
    padded_df_list = glob.glob(DATA_DIR + "\\padded*.fea")
    if (len(df_list) == 0 or not getLatest) or (len(padded_df_list) == 0 and returnPadded):
        timestamp_str = extras.nowString()
        print("Importing and processing from scratch: timestamp is " + timestamp_str)
        df_path = DATA_DIR + "\\df" + "_" + timestamp_str + ".fea"
        padded_df_path = DATA_DIR + "\\padded_df" + "_" + timestamp_str + ".fea"

        df = readFromCSV(TRAJECTORIES_DIR, threshold)
        df = postProcessing(df, IMAGING_PARAMETERS_PATH)

        if assignTrajStates:
            df = assignAllTrajStates(df, params_path=MODEL_PARAMETERS_PATH, baseLen=64.,
                                     isPar=isPar)  # this is not put into post processing
        elif "state" not in df.columns:
            df["state"] = 0.

        df.to_feather(df_path)
        if returnPadded:
            padded_df = addPadding(df, isPar)
            padded_df.to_feather(padded_df_path)
    else:
        df_list = glob.glob(DATA_DIR + "\\df*.fea")
        padded_df_list = glob.glob(DATA_DIR + "\\padded*.fea")
        df_path = max(df_list, key=os.path.getctime)
        df = pd.read_feather(df_path)
        print("Dataframe read from " + df_path)
        if returnPadded:
            padded_df_path = max(padded_df_list, key=os.path.getctime)
            padded_df = pd.read_feather(padded_df_path)
            print("Padded dataframe read from " + padded_df_path)

    if returnPadded:
        return df, padded_df
    else:
        return df


def generateDiffs(df, t_column_name="frame"):
    particles = df.particle.unique()
    gb = df.groupby("particle")
    for particle_id in particles:
        df_particle = gb.get_group(particle_id)
        t = df_particle[t_column_name].values
        diffs_mat = np.triu(
            scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(t[:, None], 'cityblock')))


def scaleUnits(df_in, imaging_params_path):
    df = df_in.copy()

    # Add time column
    columns = df.columns
    df["t"] = 0.
    # reorder columns
    frame_ind = int(np.argwhere(columns == "frame")) + 1
    columns_new = np.append(columns[np.arange(frame_ind)], "t")
    columns_new = np.append(columns_new, columns[frame_ind + np.arange(len(columns) - frame_ind)])
    df = df[columns_new]

    # Also add fps column
    df["fps"] = 1.  # default

    all_experiments = df.experiment.unique()
    N_experiments = len(all_experiments)

    try:
        parameters_table = pd.read_csv(imaging_params_path)
    except:
        return df

    for exp_id, experiment in enumerate(all_experiments):
        if experiment in parameters_table.experiment.values:
            cur_params = parameters_table[parameters_table.experiment == experiment]
            # obviously not elegant
            fps = cur_params.fps.values[0]
            micron2pix = cur_params.micron2pix.values[0]
            # Scale x,y
            exp_index = df["experiment"] == experiment
            df.loc[exp_index, "x"] = df.loc[exp_index, "x"].mul(micron2pix)
            df.loc[exp_index, "y"] = df.loc[exp_index, "y"].mul(micron2pix)
            df.loc[exp_index, "t"] = df.loc[exp_index, "frame"].div(fps)
            df.loc[exp_index, "fps"] = fps
    return df


def postProcessing(df, imaging_params_path="None"):
    df = setInitFrame(df)
    df = scaleUnits(df, imaging_params_path)
    df = removeImmobileParticle(df, imaging_params_path)  # numbers from Indrani paper
    df = addDisplacement(df, "x")
    df = addDisplacement(df, "y")
    df = addDisplacement(df, "t")  # must for bayesianTools

    df = addTrajDuration(df)

    df["dr2"] = df["dx"].pow(2) + df["dy"].pow(2)  # must for bayesianTools

    return df


def vec_translate(a, my_dict):
    # From https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    return np.vectorize(my_dict.__getitem__)(a)
