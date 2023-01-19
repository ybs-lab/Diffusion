import os
import numpy as np
import pandas as pd
import glob
from concurrent.futures import ProcessPoolExecutor
from utils import now_string, vec_translate, index_intervals_extract
from trajAnalysis import calc_rolling_R_gyration, intersecting_neighbors, assign_D_n_estimate, calculate_local_D, \
    break_trajectories_collisions, smooth_states, undersample_all_particles
from config import TRAJECTORIES_DIR, DATA_DIR, IMAGING_PARAMETERS_PATH, \
    PADDING_INTERPOLATION_MAX_FRAMES
from model import pack_model_params, model_generate_trajectories
from model_utils import GenerationMode


def import_all(get_latest=True, assign_traj_states=False, break_trajectory_at_collisions=False,
               is_parallel=False):
    df_list = glob.glob(DATA_DIR + "/df*.fea")
    if len(df_list) == 0 or not get_latest:  # read from csv and do post processing (slow)
        timestamp_str = now_string()
        print("Importing and processing from scratch: timestamp is " + timestamp_str)
        df_path = os.path.join(DATA_DIR, "df" + "_" + timestamp_str + ".fea")
        df = read_from_csv(TRAJECTORIES_DIR)
        df = post_processing(df, IMAGING_PARAMETERS_PATH, assign_traj_states=assign_traj_states,
                             undersampleLongTrajectories=True,
                             break_trajectory_at_collisions=break_trajectory_at_collisions, is_parallel=is_parallel)
        df.to_feather(df_path)
    else:  # read from feather (fast)
        df_list = glob.glob(os.path.join(DATA_DIR, "df*.fea"))
        df_path = max(df_list, key=os.path.getctime)
        df = pd.read_feather(df_path)
        print("Dataframe read from " + df_path.replace('\\', '/'))
    return df


def post_processing(df, imaging_params_path=IMAGING_PARAMETERS_PATH, assign_traj_states=False,
                    break_trajectory_at_collisions=False, undersampleLongTrajectories=True, is_parallel=False):
    print("Beginning post-processing of the dataframe...")
    # 1. Padding and interpolating missing frames
    df = add_padding(df, is_parallel=is_parallel)
    df = interpolate_padding(df, is_parallel=is_parallel)

    # 2. Scale units (fps and micron2pix)
    df = scale_units(df, imaging_params_path)

    # # 3. Handle neighbors (not for synthetic data or 1 particle trajectories)
    df = intersecting_neighbors(df, is_parallel=is_parallel)
    if ~((len(df.particle.unique()) == 1) | (df.experiment.unique()[0].lower() == "synthetic")):
        if break_trajectory_at_collisions:
            split_df_array = break_trajectories_collisions(df, is_parallel=is_parallel)
            df = split_particle_trajectories(split_df_array)

    # 4. Undersample long trajectories
    if undersampleLongTrajectories:
        df = undersample_all_particles(df, is_parallel=is_parallel)

    # 5. Add useful and simple information:
    df = set_init_frame(df)
    df = label_immobile_particle(df, imaging_params_path)  # this is for totally stuck particles
    df = add_displacement(df, "x")
    df = add_displacement(df, "y")
    df = add_displacement(df, "t")  # must for bayesianTools
    df = add_traj_duration(df)
    df["dr2"] = df["dx"].pow(2) + df["dy"].pow(2)  # must for bayesianTools

    # 6. Add more advanced statistics
    df = calc_rolling_R_gyration(df, is_parallel=is_parallel)

    if assign_traj_states:  # This is the long bayesian stuff
        pass
        # df = labelAllTrajectories(df, is_parallel=is_parallel)
        # df = smooth_states(df, is_parallel=is_parallel)
    elif "state" not in df.columns:
        df["state"] = 0

    df = assign_D_n_estimate(df, is_parallel=is_parallel)
    df = calculate_local_D(df, T=30, is_parallel=is_parallel)
    df = calculate_local_D(df, T=90, is_parallel=is_parallel)
    df = calculate_local_D(df, T=150, is_parallel=is_parallel)

    print("Done with post-processing!")
    return df


def generate_diffuse_tether_trajectories(T_stick, T_unstick, D, A, dt=1. / 30., N_steps=1024, N_particle=1, init_S=0,
                                         do_post_processing=True, undersample_ratio=0, save_files=False,
                                         generation_mode=GenerationMode.DONT_FORCE,
                                         is_parallel=False):
    """
    Generate a trajectory (see Model class) and wrap it as a dataframe
    """
    model_params = pack_model_params(T_stick, T_unstick, D, A, dt)
    states_arr, X_arr, X_tether_arr = model_generate_trajectories(N_steps, N_particle, init_S, model_params,
                                                                  generation_mode)

    # Here model ends and it's just pandas stuff
    frame = np.tile(np.arange(N_steps, dtype=int), [N_particle, 1]).flatten()
    particle = np.tile(np.arange(N_particle, dtype=int), [N_steps, 1]).T.flatten()
    x = X_arr[:, :, 0].flatten()
    y = X_arr[:, :, 1].flatten()
    state = states_arr.flatten().astype(int)
    x_tether = X_tether_arr[:, :, 0].flatten()
    y_tether = X_tether_arr[:, :, 1].flatten()
    mass = 0. * frame
    df = pd.DataFrame.from_dict({
        'frame': frame,
        'particle': particle,
        'x': x,
        'y': y,
        'mass': mass,
        'state': state,
        'x_tether': x_tether,
        'y_tether': y_tether,
    })

    # TBD FIX THIS TO SOMETHING PER PARTICLE
    N_samp = N_particle * N_steps
    N_samples_to_discard = int(np.floor(N_samp * undersample_ratio))
    random_index = np.random.choice(range(N_samp), N_samples_to_discard, replace=False)
    mask_array = np.ones(N_samp, dtype=bool)
    mask_array[random_index] = False
    df = df[mask_array].reset_index()

    timestamp_str = now_string()
    df["experiment"] = "synthetic"
    df["filename"] = "synth_" + timestamp_str
    df["file_particle_id"] = df["particle"]
    df["video_frame"] = df["frame"]

    if do_post_processing and (undersample_ratio == 0.):
        df = post_processing(df, is_parallel=is_parallel, assign_traj_states=False,
                             break_trajectory_at_collisions=False,
                             undersampleLongTrajectories=False)
    else:  # just do a very minimal post_processing
        df = scale_units(df, None,fps=1/dt)
        df = add_displacement(df, "x")
        df = add_displacement(df, "y")
        df = add_displacement(df, "t")  # must for bayesianTools
        df = add_traj_duration(df)
        df['isNotPad'] = True

    if save_files:
        # save now
        filepath = "./Trajectories/Synthetic/"
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        df.to_csv(path_or_buf=os.path.join(filepath, "synth_" + timestamp_str + ".csv"))

    return df


def read_from_csv(traj_path):
    # Return a pandas dataframe with all the experimental data found in input path.
    # In directory traj_path, each different folder is considered a different experiment
    # threshold is related to breaking of long trajectories, not recommended to use!
    #
    # Notes:
    # Multiple files are concatenated and particle IDs are renamed such that all particle IDs are unique
    # Also drop all columns except permitted ones

    allowed_columns = ['particle', 'frame', 'x', 'y', 'mass']  # in the desired order
    offset = 0
    experiments = sorted(os.listdir(traj_path))
    if len(experiments) == 0:
        print(("No folders found in directory: " + traj_path).replace('\\', '/'))
        df = []
        return

    for n_exp, experiment in enumerate(experiments):
        fullpath = os.path.join(traj_path, experiment)
        files = sorted(os.listdir(fullpath))
        if (len(files) == 0) and not (experiment == 'Synthetic'):
            print(("No files found in directory: " + fullpath).replace('\\', '/'))
            continue
        else:
            for n, file in enumerate(files):
                cur_df = pd.read_csv(os.path.join(fullpath, file))
                cur_df.drop(cur_df.columns.difference(allowed_columns), axis=1,
                            inplace=True)  # remove unwanted columns
                cur_df = cur_df[allowed_columns]  # reorder columns
                cur_df["particle"] = cur_df["particle"].astype('int')  # convert to int
                cur_df["experiment"] = experiment
                cur_df["filename"] = file
                cur_df["video_frame"] = cur_df["frame"]  # this is important for later
                # make sure the dataframe is well sorted
                cur_df = cur_df.sort_values(["particle", "frame"])
                cur_df.index = range(len(cur_df))
                N_particles = rename_particle_id(cur_df, offset)
                offset = offset + N_particles
                if "df" not in locals():
                    df = cur_df
                else:
                    df = df.append(cur_df, "ignore_index")  # add new df on top of existing df
    return df


def rename_particle_id(df, offset):
    # Rename particle IDs because different files might have the same particle ID (but the particles are different)
    # This alters the input dataframe!
    particleIDs = pd.unique(df.particle)
    Nparticles = len(particleIDs)
    dict_ID_to_index = dict(zip(particleIDs, offset + np.arange(Nparticles)))
    df["file_particle_id"] = df.particle  # keep original particle id for tracing purpose
    df.particle = vec_translate(df.particle, dict_ID_to_index)
    return Nparticles


def set_init_frame(df):
    # Make sure the initial frame of each particle is 0
    df['init_frame'] = df.groupby('particle')['frame'].transform('first')
    df['frame'] = df['frame'] - df['init_frame']
    return df


def add_displacement(df, ax):
    # ax = "x" or "y" or "t"
    # Add time column
    columns = df.columns
    diff_ax_name = "d" + ax
    if len(np.argwhere(columns == diff_ax_name)) == 0:  # no displacement column yet
        copy_df = df[["particle", ax]].copy()
        df[diff_ax_name] = copy_df.groupby("particle").diff()  # this makes sure there is NaN at interparticle rows
        # reorder columns
        ax_ind = int(np.argwhere(columns == ax)) + 1
        columns_new = np.append(columns[np.arange(ax_ind)], diff_ax_name)
        columns_new = np.append(columns_new, columns[ax_ind + np.arange(len(columns) - ax_ind)])
        df = df[columns_new]
    else:
        df[diff_ax_name] = df.groupby("particle")[ax].diff()  # this makes sure there is NaN at interparticle rows
    return df


def label_immobile_particle(df, imaging_params_path):
    # t_record in sec, cutoff_dist in micrometers
    df["isMobile"] = True
    try:
        parameters_table = pd.read_csv(imaging_params_path)
    except:
        return df

    df_experiments = df.experiment.unique()
    table_experiments = parameters_table.experiment
    parameters_table = parameters_table[np.isin(table_experiments, df_experiments)]
    allowed_experiments = parameters_table[parameters_table.Remove_Immobile].experiment.values

    t_record_list = parameters_table[parameters_table.Remove_Immobile].Remove_Immobile_t_record.values
    cutoff_dist_list = parameters_table[parameters_table.Remove_Immobile].Remove_Immobile_cutoff_dist.values

    D = df.copy()[["experiment", "particle", "t", "x", "y"]]
    D.loc[:, "isMobile"] = True
    particles = D.particle.unique()
    # this is not really msd but just (r(t)-r(0))^2
    # D['MSD'] = (D['x'] - D.groupby('particle')['x'].transform('first')).pow(2) + (
    #         D['y'] - D.groupby('particle')['y'].transform('first')).pow(2)
    gb_exp = df.groupby("experiment")
    print("Labeling immobile particles...")
    for n_iter, experiment in enumerate(allowed_experiments):
        df_exp = gb_exp.get_group(experiment)
        particles = df_exp.particle.unique()
        G = df_exp.groupby("particle")
        for particle_id in particles:
            particle_df = G.get_group(particle_id)
            t_record = t_record_list[n_iter]
            cutoff_dist = cutoff_dist_list[n_iter]
            x = particle_df.x.values
            y = particle_df.y.values
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            dx = x - x[0]
            dy = y - y[0]
            # if no more than 3 frames saw the particle escaping the cutoff distance then it's stuck (2 frames to account for weird jitters)
            if np.sum((dx ** 2 + dy ** 2 > cutoff_dist).astype(int)) < 4:
                D.loc[particle_df.index, "isMobile"] = False
            # dont need something  so complicated
            # particle_df_trimmed = particle_df[particle_df["t"] <= t_record]
            # x = particle_df_trimmed.x.values
            # y = particle_df_trimmed.y.values
            # x = x[~np.isnan(x)]
            # y = y[~np.isnan(y)]
            # dx = scipy.spatial.distance.pdist(x[:, None], 'cityblock')
            # dy = scipy.spatial.distance.pdist(y[:, None], 'cityblock')
            # if np.max(dx ** 2 + dy ** 2) < cutoff_dist ** 2:
            #     D.loc[particle_df.index, "isMobile"] = False

    # return df[D["isMobile"]].reset_index(drop=True)
    df["isMobile"] = D["isMobile"]
    return df


def add_traj_duration(df):  # in frames
    df = df.copy()
    df["trajDuration"] = 0.
    df["trajDurationSec"] = 0.
    len_array = df.groupby("particle").apply(len)
    len_dict = len_array.to_dict()
    fps_array = df.groupby("particle").fps.first()
    len_dict = len_array.to_dict()
    fps_dict = fps_array.to_dict()
    # groupby experiment to avoid memory issues
    experiments = df["experiment"].unique()
    gb_exp = df.groupby("experiment")
    for n, experiment in enumerate(experiments):
        df_exp = gb_exp.get_group(experiment)
        len_df_exp = len(df_exp)
        indices = df_exp.index
        particles = df_exp["particle"].values
        trajDuration = vec_translate(particles, len_dict)  # use basic vectorized dictionary function
        trajDurationSec = trajDuration / vec_translate(particles, fps_dict)
        df.loc[df_exp.index, "trajDuration"] = trajDuration
        df.loc[df_exp.index, "trajDurationSec"] = trajDurationSec

    return df


def get_missing_frames(particle_df):
    frames = particle_df.frame.values
    min_frame = np.min(frames)
    max_frame = np.max(frames)
    missingFramesInds = np.where(~np.isin(np.arange(min_frame, max_frame), frames))[0]
    missingFrames = np.arange(min_frame, max_frame)[missingFramesInds]
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


def add_padding(df, is_parallel=False):
    df = df.copy()
    df["isNotPad"] = True
    particles = df.particle.unique()
    gb = df.groupby("particle")
    all_particle_df = [gb.get_group(i) for i in particles]
    missingFrame_df_array = [pd.DataFrame([]) for i in particles]

    print("Padding the dataframe...")
    if is_parallel:
        with ProcessPoolExecutor() as executor:
            for j, output in enumerate(executor.map(get_missing_frames, all_particle_df)):
                missingFrame_df_array[j] = output
                # print("Padded particle " + str(j) + " (Parallel)")
    else:
        for j, particle in enumerate(particles):
            output = get_missing_frames(all_particle_df[j])
            missingFrame_df_array[j] = output
            # print("Padded particle " + str(j))

    df_to_append = pd.concat(missingFrame_df_array, ignore_index=True)
    df_to_append.index = df_to_append.index + df.index.max() + 1
    df = df.append(df_to_append)
    df = df.sort_values(["particle", "frame"])
    df = df.reset_index(drop=True)
    return df


def interpolate_missing_frames(particle_df, nan_threshold=PADDING_INTERPOLATION_MAX_FRAMES):
    particle_df = particle_df.copy()
    padding_segments = list(index_intervals_extract(particle_df.index[~particle_df.isNotPad]))
    first_index = particle_df.index[0]
    last_index = particle_df.index[-1]
    for pad_interval in padding_segments:
        len_pad = pad_interval[1] - pad_interval[0] + 1
        if len_pad < nan_threshold:
            if (pad_interval[0] > first_index) and (pad_interval[1]) < last_index:
                new_x = np.linspace(particle_df.x[pad_interval[0] - 1], particle_df.x[pad_interval[1] + 1], 2 + len_pad)
                new_y = np.linspace(particle_df.y[pad_interval[0] - 1], particle_df.y[pad_interval[1] + 1], 2 + len_pad)
                new_mass = np.linspace(particle_df.mass[pad_interval[0] - 1], particle_df.mass[pad_interval[1] + 1],
                                       2 + len_pad)
                particle_df.loc[pad_interval[0]:pad_interval[1], "x"] = new_x[1:-1]
                particle_df.loc[pad_interval[0]:pad_interval[1], "y"] = new_y[1:-1]
                particle_df.loc[pad_interval[0]:pad_interval[1], "mass"] = new_mass[1:-1]

    return particle_df


def interpolate_padding(df, is_parallel=False):
    df = df.copy()
    particles = df.particle.unique()
    gb = df.groupby("particle")
    all_particle_df = [gb.get_group(i) for i in particles]
    interpolated_df_array = [pd.DataFrame([]) for i in particles]

    print("Interpolating the dataframe's padding...")
    if is_parallel:
        with ProcessPoolExecutor() as executor:
            for j, output in enumerate(executor.map(interpolate_missing_frames, all_particle_df)):
                interpolated_df_array[j] = output
                # print("Interpolated padding for particle " + str(j) + " (Parallel)")
    else:
        for j, particle in enumerate(particles):
            output = interpolate_missing_frames(all_particle_df[j])
            interpolated_df_array[j] = output
            # print("Interpolated padding for particle " + str(j))

    df = pd.concat(interpolated_df_array, ignore_index=True).sort_index()
    # df = df.reset_index(drop=True)
    return df


def scale_units(df_in, imaging_params_path,fps=None):

    default_fps = 30.
    default_micron2pix = 0.097

    if df_in.experiment.unique()[0].lower() == "synthetic":
        default_micron2pix = 1.
        if fps is not None:
            default_fps = fps

    df = df_in.copy()

    # Add time column
    columns = df.columns
    if 't' not in columns:
        df["t"] = df["frame"]
        # reorder columns
        frame_ind = int(np.argwhere(columns == "frame")) + 1
        columns_new = np.append(columns[np.arange(frame_ind)], "t")
        columns_new = np.append(columns_new, columns[frame_ind + np.arange(len(columns) - frame_ind)])
        df = df[columns_new]

    # Also add fps column
    if 'fps' not in columns:
        df["fps"] = default_fps
    if 'micron2pix' not in columns:
        df["micron2pix"] = default_micron2pix

    all_experiments = df.experiment.unique()
    N_experiments = len(all_experiments)

    try:
        parameters_table = pd.read_csv(imaging_params_path)
        for exp_id, experiment in enumerate(all_experiments):
            if experiment in parameters_table.experiment.values:
                cur_params = parameters_table[parameters_table.experiment == experiment]
                # obviously not elegant
                fps = cur_params.fps.values[0]
                micron2pix = cur_params.micron2pix.values[0]
                # Scale x,y
                exp_index = df["experiment"] == experiment
                df.loc[exp_index, "fps"] = fps
                df.loc[exp_index, "micron2pix"] = micron2pix
    except:
        pass

    df.x = df.x * df.micron2pix
    df.y = df.y * df.micron2pix
    df.t = df.frame / df.fps

    return df


def split_particle_trajectories(split_particle_df_array):
    count = 0
    splitted_df_array = np.empty(len(split_particle_df_array), dtype=object)
    for i, split_df_array in enumerate(split_particle_df_array):
        for new_particle_df in split_df_array:
            new_particle_df.loc[:, "particle"] = count
            count += 1
        if len(split_df_array) > 0:
            splitted_df_array[i] = pd.concat(split_df_array)
    df = pd.concat(splitted_df_array).reset_index(drop=True)
    return df
