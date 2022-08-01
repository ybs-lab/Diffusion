import numpy as np
from numpy import random
import pandas as pd
import os
from importData import postProcessing
from extras import nowString, generate_CTMC_Matrix


def generateDiffuseTetherModel(T_stick, T_unstick, D, A, N_steps=1024, dt=1. / 30., N_particle=4, init_state=0,
                               undersample_ratio=0, saveFiles=False, forceOneMode=0, isPar=False):
    if forceOneMode == "free":
        init_state = 0.
    elif forceOneMode == "stuck":
        init_state = 1.

    init_state_arr = np.zeros(N_particle)
    if init_state == 1:
        init_state_arr += 1
    elif init_state == "random":
        init_state_arr = np.random.randint(2, size=N_particle).astype(float)

    states_arr = np.zeros([N_particle, N_steps])
    X_arr = np.zeros([N_particle, N_steps, 2])
    X_tether_arr = np.zeros([N_particle, N_steps, 2])
    states_arr[:, 0] = init_state_arr

    P = generate_CTMC_Matrix(T_stick, T_unstick, dt)  # markov transition matrix
    if forceOneMode == "free":
        P[:, 0] = 0.
        P[:, 1] = 1.
    elif forceOneMode == "stuck":
        P[:, 0] = 1.
        P[:, 1] = 0.

    # Stream of 2D gaussian RV with variance 1
    gaussian_Stream = random.default_rng().normal(loc=0.0, scale=np.sqrt(1 / 2), size=[N_particle, N_steps, 2])

    # This is for when a particle sticks and another random sample is needed for the tether point
    extra_gaussian_Stream = random.default_rng().normal(loc=0.0, scale=np.sqrt(1 / 2), size=[N_particle, N_steps, 2])

    uniform_Stream = random.default_rng().random(size=[N_particle, N_steps])

    X_tether_arr[np.where(init_state_arr == 1.), 0, :] = np.sqrt(A) * gaussian_Stream[np.where(init_state_arr == 1.), 0,
                                                                      :]  # for clarity X_tether is initialized only for stuck

    # note: this is valid only when dt<<T_stick,T_unstick

    for n in range(1, N_steps):
        free_inds = np.where(states_arr[:, n - 1] == 0.)[0]
        stuck_inds = np.where(states_arr[:, n - 1] == 1.)[0]

        # Free particles diffuse
        X_arr[free_inds, n, :] = X_arr[free_inds, n - 1, :] + np.sqrt(4 * D * dt) * gaussian_Stream[free_inds, n, :]
        # Stuck particles wiggle
        X_arr[stuck_inds, n, :] = X_tether_arr[stuck_inds, n - 1, :] + np.sqrt(A) * gaussian_Stream[stuck_inds, n, :]

        # Tether point continues UNLESS going to stick
        X_tether_arr[:, n, :] = X_tether_arr[:, n - 1, :]

        # Stick or unstick:
        # print(P[0,0])
        # print(free_inds)
        sticking_inds = free_inds[np.where(uniform_Stream[free_inds, n] > P[0, 0])[0]]
        staying_free_inds = np.setdiff1d(free_inds, sticking_inds)
        unsticking_inds = stuck_inds[np.where(uniform_Stream[stuck_inds, n] > P[1, 1])[0]]
        staying_stuck_inds = np.setdiff1d(stuck_inds, unsticking_inds)

        states_arr[np.union1d(unsticking_inds, staying_free_inds), n] = 0.
        states_arr[np.union1d(sticking_inds, staying_stuck_inds), n] = 1.

        # Sticking particles tether to a point
        reduce_tethering_range = 1e-10
        X_tether_arr[sticking_inds, n, :] = X_arr[sticking_inds, n, :] + \
                                            reduce_tethering_range*np.sqrt(A) * extra_gaussian_Stream[
                                                                                      sticking_inds, n, :]
    frame = np.tile(np.arange(N_steps, dtype=int), [N_particle, 1]).flatten()
    particle = np.tile(np.arange(N_particle, dtype=int), [N_steps, 1]).T.flatten()
    x = X_arr[:, :, 0].flatten()
    y = X_arr[:, :, 1].flatten()
    state = states_arr.flatten()
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

    timestamp_str = nowString()
    df["experiment"] = "synthetic"
    df["filename"] = "synth_" + timestamp_str
    df["file_particle_id"] = df["particle"]
    df["video_frame"] = df["frame"]
    df = postProcessing(df, isPar=isPar, assignTrajStates=False, breakTrajectoryAtCollisions=False,
                        undersampleLongTrajectories=False)

    if saveFiles:
        # save now
        filepath = "./Trajectories/Synthetic/"
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        df.to_csv(path_or_buf=os.path.join(filepath, "synth_" + timestamp_str + ".csv"))

    return df


def generateDiffuseTetherModelOld(T_stick, T_unstick, D, A, dt=1 / 30, T_end=2.0, N_particle=4, init_state=0,
                                  undersample_ratio=0, saveFiles=False, forceOneMode=0, isPar=False):
    init_state_arr = np.zeros(N_particle)
    if init_state == 1:
        init_state_arr += 1
    elif init_state == "random":
        init_state_arr = np.random.randint(2, size=N_particle).astype(float)

    for particle_id in range(N_particle):
        particle_init_state = init_state_arr[particle_id]
        largeN = 100
        t_stick_rolls = random.default_rng().exponential(T_stick, largeN) + dt  # minimum of dt for transitions
        t_unstick_rolls = random.default_rng().exponential(T_unstick, largeN) + dt  # minimum of dt for transitions
        all_t_rolls = np.zeros(2 * largeN)
        if not particle_init_state:  # start free
            all_t_rolls[::2] = t_stick_rolls
            all_t_rolls[1::2] = t_unstick_rolls
        else:  # start stuck
            all_t_rolls[::2] = t_unstick_rolls
            all_t_rolls[1::2] = t_stick_rolls
        t_stick_unstick = np.cumsum(all_t_rolls)
        t_stick_unstick = t_stick_unstick[t_stick_unstick < T_end - dt]
        t_stick_unstick_with_T_end = np.zeros(len(t_stick_unstick) + 1)
        t_stick_unstick_with_T_end[:-1] = t_stick_unstick
        t_stick_unstick_with_T_end[-1] = T_end
        if forceOneMode != 0:
            t_stick_unstick_with_T_end = [T_end]  # important for only stuck

        t = np.arange(0.0, T_end + dt, step=dt)
        N_samp = len(t)
        r = np.zeros([N_samp, 2])
        state_array = np.zeros(N_samp)
        r0 = 1000 * (random.rand(2) - 0.5)
        r[0, :] = r0
        state_array[0] = 0.5  # init_state
        prev_t = 0
        prev_position = r0
        cur_state = particle_init_state
        for t_switch in t_stick_unstick_with_T_end:
            ind = (t > prev_t) & (t <= t_switch)
            N_steps = sum(ind)
            if (forceOneMode == 0 and not cur_state) or forceOneMode == 1:  # free
                # step_angles = random.default_rng().uniform(0, 2 * np.pi, N_steps)
                steps_x = random.default_rng().normal(loc=0.0, scale=np.sqrt(2 * D * dt), size=N_steps)
                steps_y = random.default_rng().normal(loc=0.0, scale=np.sqrt(2 * D * dt), size=N_steps)
                x = prev_position[0] + np.cumsum(steps_x)
                y = prev_position[1] + np.cumsum(steps_y)
            elif (forceOneMode == 0 and cur_state) or forceOneMode == 2:  # stuck
                pos_x = random.default_rng().normal(loc=0.0, scale=np.sqrt(A / 2), size=N_steps)
                pos_y = random.default_rng().normal(loc=0.0, scale=np.sqrt(A / 2), size=N_steps)
                x = prev_position[0] + pos_x
                y = prev_position[1] + pos_y
            else:  # cur_state is 0 or 1
                pass
            r[ind, 0] = x
            r[ind, 1] = y
            if forceOneMode == 0:
                state_array[ind] = cur_state
            elif forceOneMode == 1:
                state_array[ind] = 0
            elif forceOneMode == 2:
                state_array[ind] = 1

            # state_array[np.where(ind)[0].min()] = 0.5  # transitions are special
            state_array[np.where(ind)[0].max()] = 0.5  # transitions are special
            prev_t = t_switch
            prev_position = np.array([x[-1], y[-1]])
            cur_state = 1 - cur_state  # switch mode

        frame = (t / dt).round().astype(int)
        x_arr = r[:, 0]
        y_arr = r[:, 1]
        mass = np.zeros(N_samp)
        particle = (particle_id * np.ones(N_samp)).astype(int)

        cur_df = pd.DataFrame({"particle": particle,
                               "frame": frame,
                               "x": x_arr,
                               "y": y_arr,
                               "mass": mass,
                               "state": state_array})

        N_samples_to_discard = int(np.floor(N_samp * undersample_ratio))
        random_index = np.random.choice(range(N_samp), N_samples_to_discard, replace=False)
        mask_array = np.ones(N_samp, dtype=bool)
        mask_array[random_index] = False
        cur_df = cur_df[mask_array]

        if "df" not in locals():
            df = cur_df
        else:
            df = df.append(cur_df, "ignore_index")  # add new df on top of existing df

    timestamp_str = nowString()
    df["experiment"] = "synthetic"
    df["filename"] = "synth_" + timestamp_str
    df["file_particle_id"] = df["particle"]
    df["video_frame"] = df["frame"]
    df = postProcessing(df, isPar=isPar, assignTrajStates=False, breakTrajectoryAtCollisions=False,
                        undersampleLongTrajectories=False)

    if saveFiles:
        # save now
        filepath = "./Trajectories/Synthetic/"
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        df.to_csv(path_or_buf=os.path.join(filepath, "synth_" + timestamp_str + ".csv"))

    return df
