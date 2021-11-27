import os
import numpy as np
import pandas as pd
import scipy.spatial
from bayesianTools import segmentTraj, particleLikelihood
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor


def renameParticleID(df, offset):
    # Rename particle IDs because different files might have the same particle ID (but the particles are different)
    # This alters the input dataframe!
    particleIDs = pd.unique(df.particle)
    Nparticles = len(particleIDs)
    dict_ID_to_index = dict(zip(particleIDs, offset + np.arange(Nparticles)))
    df.particle = df.particle.replace(dict_ID_to_index)
    return Nparticles


def readFromCSV(P1, P2, salinity, threshold=100):
    # Insert only tuples to this function (for now)
    # Read csv files with trajectories from local folder (P1 on P2 with specific salinity)
    # Multiple files are concatenated and particle IDs are renamed
    # Also drop all columns except permitted ones

    allowed_columns = ['particle', 'frame', 'x', 'y', 'mass']  # in the desired order
    offset = 0
    for i in range(len(P1)):
        for j in range(len(P2)):
            for k in range(len(salinity)):
                file_dir = os.getcwd() + "/Trajectories/" + P1[i] + "on" + P2[j] + "/Salinity_" + str(
                    salinity[k]) + "/"
                if not (os.path.isdir(file_dir)):
                    # print("Directory not found: " + file_dir)  #commented this because no 60/110 salinity for GonK
                    continue
                files = os.listdir(file_dir)
                if len(files) == 0:
                    print("No files found in directory: " + file_dir)
                    continue
                else:
                    for n in range(len(files)):
                        cur_df = pd.read_csv(file_dir + files[n])
                        cur_df.drop(cur_df.columns.difference(allowed_columns), axis=1,
                                    inplace=True)  # remove unwanted columns
                        cur_df = cur_df[allowed_columns]  # reorder columns
                        cur_df["particle"] = cur_df["particle"].astype('int')  # convert to int
                        cur_df["peptides"] = np.full(len(cur_df), P1[i] + " on " + P2[j])  # add peptides
                        cur_df["salinity"] = np.full(len(cur_df), salinity[k], dtype=int)  # add salinity
                        cur_df["experiment"] = np.full(len(cur_df), P1[i] + " on " + P2[j] + " ; s=" + str(
                            salinity[k]))  # add salinity

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


def addTime(df, fps):
    # Add time column
    columns = df.columns
    if len(np.argwhere(columns == "t")) == 0:  # no time column yet
        df["t"] = df["frame"].div(fps)
        # reorder columns
        frame_ind = int(np.argwhere(columns == "frame")) + 1
        columns_new = np.append(columns[np.arange(frame_ind)], "t")
        columns_new = np.append(columns_new, columns[frame_ind + np.arange(len(columns) - frame_ind)])
        df = df[columns_new]
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


def removeImmobileParticle(df, t_record, cutoff_dist, excluded_experiments=[]):
    # t_record in sec, cutoff_dist in micrometers
    D = df.copy()[["experiment", "particle", "t", "x", "y"]]
    D.loc[:, "keepIndex"] = True
    particles = D.particle.unique()
    # this is not really msd but just (r(t)-r(0))^2
    # D['MSD'] = (D['x'] - D.groupby('particle')['x'].transform('first')).pow(2) + (
    #         D['y'] - D.groupby('particle')['y'].transform('first')).pow(2)
    G = D.groupby("particle")
    for particle_id in particles:
        particle_df = G.get_group(particle_id)
        if particle_df["experiment"].unique() not in excluded_experiments:
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
    df["trajDuration"] = 0
    particles = df.particle.unique()
    len_array = df.groupby("particle").apply(len)
    gb = df.groupby("particle")
    for particle_id in particles:
        ind = gb.get_group(particle_id).index
        df.loc[ind, "trajDuration"] = len_array[particle_id]
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


def assignAllTrajStates(df_in, baseLen=64., isPar=False):
    file_dir = os.getcwd() + "/Trajectories/" + "model_parameters_table.csv"
    parameters_table = pd.read_csv(file_dir)

    df = df_in.copy()
    all_experiments = df.experiment.unique()
    N_experiments = len(all_experiments)
    df["state"] = 0.
    df["logL_states"] = 0.
    gb_exp = df.groupby("experiment")

    for exp_id,experiment in enumerate(all_experiments):
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
                    print("Assigned states to particle " + str(j) + " of " + str(N_particles) + " ; Experiment "+str(exp_id)+"/"+str(N_experiments))
        else:
            for j, particle in enumerate(particles):
                output = assignParticleTrajState(all_particle_df[j], T_stick, T_unstick, D, A, baseLen=baseLen)
                df.loc[output[0], "state"] = output[1]
                df.loc[output[0], "logL_states"] = output[2]
                print("Assigned states to particle " + str(j) + " of " + str(N_particles) + " ; Experiment "+str(exp_id)+"/"+str(N_experiments))

        print("Assigned states to experiment " + experiment)

    return df


def importAll(threshold=1000000, isPar=False):
    default_file_name = "mydata.fea"
    if os.path.isfile(os.getcwd() + "/" + default_file_name):
        df = pd.read_feather(os.getcwd() + "/" + default_file_name)
    else:
        fps = 50.
        P1 = ["G", "K", "R", "L"]
        P2 = ["G", "K", "W", "B"]
        # P1 = ["Z"]
        # P2 = ["Z"]
        salinity = range(161)
        # salinity = [0, 60, 110, 160]
        df = readFromCSV(P1, P2, salinity, threshold)
        df = postProcessing(df, fps, isPar=isPar)
        saveData(df, default_file_name, 0)
    return df


def saveData(df, name, mode):
    # Save data in feather format, TBD other formats + save data with date/time etc.
    save_path = os.getcwd() + "/" + name
    df.to_feather(save_path)


def postProcessing(df, fps, isPar=False):
    df.loc[df["experiment"] == "R on W ; s=0", "experiment"] = "RW"  # rename
    df.loc[df["experiment"] == "L on B ; s=0", "experiment"] = "Lab"  # rename
    df = setInitFrame(df)
    df = addTime(df, fps)

    df.loc[df["peptides"] != "R on W", "x"] = df.loc[df["peptides"] != "R on W", "x"].mul(
        0.29)  # rescale pix to microns
    df.loc[df["peptides"] != "R on W", "y"] = df.loc[df["peptides"] != "R on W", "y"].mul(
        0.29)  # rescale pix to microns

    df.loc[df["experiment"] == "Lab", "t"] = df.loc[df["experiment"] == "Lab", "t"].mul(fps / 20.)  # rescale time
    df.loc[df["experiment"] == "Lab", "x"] = df.loc[df["experiment"] == "Lab", "x"].mul(
        10 / 85 / 0.29)  # rescale pix to microns
    df.loc[df["experiment"] == "Lab", "y"] = df.loc[df["experiment"] == "Lab", "y"].mul(
        10 / 85 / 0.29)  # rescale pix to microns
    df = removeImmobileParticle(df, t_record=0.4, cutoff_dist=1.2 * 0.29,
                                excluded_experiments=["RW", "Lab"])  # numbers from Indrani paper
    df = addDisplacement(df, "x")
    df = addDisplacement(df, "y")
    df = addDisplacement(df, "t")  # must for bayesianTools

    df = addTrajDuration(df)
    df["dr2"] = df["dx"].pow(2) + df["dy"].pow(2)  # must for bayesianTools

    df = assignAllTrajStates(df, baseLen=64., isPar=isPar)

    return df
