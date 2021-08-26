import os
import numpy as np
import pandas as pd


def renameParticleID(df, offset):
    # Rename particle IDs because different files might have the same particle ID (but the particles are different)
    # This alters the input dataframe!
    particleIDs = pd.unique(df.particle)
    Nparticles = len(particleIDs)
    dict_ID_to_index = dict(zip(particleIDs, offset + np.arange(Nparticles)))
    df.particle = df.particle.replace(dict_ID_to_index)
    return Nparticles


def readFromCSV(P1, P2, salinity):
    # Insert only tuples to this function (for now)
    # Read csv files with trajectories from local folder (P1 on P2 with specific salinity)
    # Multiple files are concatenated and particle IDs are renamed
    # Also drop all columns except permitted ones

    allowed_columns = ['particle', 'frame', 'x', 'y', 'mass']  # in the desired order
    offset = 0
    for i in range(len(P1)):
        for j in range(len(P2)):
            for k in range(len(salinity)):
                file_dir = os.getcwd() + "\\Trajectories\\" + P1[i] + "on" + P2[j] + "\\Salinity_" + str(
                    salinity[k]) + "\\"
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
                        cur_df.drop(cur_df.columns.difference(allowed_columns), 1,
                                    inplace=True)  # remove unwanted columns
                        cur_df = cur_df[allowed_columns]  # reorder columns
                        cur_df["particle"] = cur_df["particle"].astype('int')  # convert to int
                        cur_df["peptides"] = np.full(len(cur_df), P1[i] + " on " + P2[j])  # add peptides
                        cur_df["salinity"] = np.full(len(cur_df), salinity[k], dtype=int)  # add salinity
                        cur_df["experiment"] = np.full(len(cur_df), P1[i] + " on " + P2[j] + " ; s=" + str(
                            salinity[k]))  # add salinity
                        N_particles = renameParticleID(cur_df, offset)
                        offset = offset + N_particles
                        if (n + i + j + k) == 0:
                            df = cur_df
                        else:
                            df = df.append(cur_df, "ignore_index")  # add new df on top of existing df

    return df


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
    # ax = "x" or "y"
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

def removeImmobileParticle(df, t_record, cutoff_dist):
    # t_record in sec, cutoff_dist in micrometers
    D = df.copy()[["particle", "t", "x", "y"]]
    D.loc[:, "keepIndex"] = True
    particles = D.particle.unique()
    #this is not really msd but just (r(t)-r(0))^2
    D['MSD'] = (D['x'] - D.groupby('particle')['x'].transform('first')).pow(2) + (
                D['y'] - D.groupby('particle')['y'].transform('first')).pow(2)
    G = D.groupby("particle")
    for particle_id in particles:
        x = G.get_group(particle_id)
        if x[x["t"] <= t_record].MSD.max() < (cutoff_dist ** 2):
            D.loc[x.index, "keepIndex"] = False

    return df[D["keepIndex"]].reset_index()

def importAll():
    default_file_name = "mydata.fea"
    if os.path.isfile(os.getcwd() + "\\" + default_file_name):
        df = pd.read_feather(os.getcwd() + "\\" + default_file_name)
    else:
        fps = 50
        P1 = ["G", "K"]
        P2 = ["G", "K"]
        salinity = [0, 60, 110, 160]
        df = readFromCSV(P1, P2, salinity)
        df = setInitFrame(df)
        df = addTime(df, fps)
        df = removeImmobileParticle(df, t_record=0.4, cutoff_dist=1.2)  # numbers from Indrani paper
        df = addDisplacement(df, "x")
        df = addDisplacement(df, "y")
        saveData(df, default_file_name, 0)
    return df


def saveData(df, name, mode):
    # Save data in feather format, TBD other formats + save data with date/time etc.
    save_path = os.getcwd() + "\\" + name
    df.to_feather(save_path)
