import numpy as np
from numpy import random
import pandas as pd
import os
from importData import postProcessing


def generateDiffuseTetherModel(T_stick, T_unstick, D, A, dt=0.02, T_end=2.0, N_particle=4, init_state=0,
                               undersample_ratio=0, experiment_name=[], salinity_index=1, saveFiles=False,
                               forceOneMode=0):
    for particle_id in range(N_particle):
        # init_state = int(np.floor(2 * random.rand()))
        largeN = 100
        t_stick_rolls = random.default_rng().exponential(T_stick, largeN) + dt  # minimum of dt for transitions
        t_unstick_rolls = random.default_rng().exponential(T_unstick, largeN) + dt  # minimum of dt for transitions
        all_t_rolls = np.zeros(2 * largeN)
        if not init_state:  # start free
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
        state_array[0] = 0.5#init_state
        prev_t = 0
        prev_position = r0
        cur_state = init_state
        for t_switch in t_stick_unstick_with_T_end:
            ind = (t > prev_t) & (t <= t_switch)
            N_steps = sum(ind)
            if (forceOneMode==0 and not cur_state) or forceOneMode == 1:  # free
                #step_angles = random.default_rng().uniform(0, 2 * np.pi, N_steps)
                steps_x =  random.default_rng().normal(loc=0.0, scale=np.sqrt(2 * D * dt), size=N_steps)
                steps_y =  random.default_rng().normal(loc=0.0, scale=np.sqrt(2 * D * dt), size=N_steps)
                x = prev_position[0] + np.cumsum(steps_x)
                y = prev_position[1] + np.cumsum(steps_y)
            elif (forceOneMode==0 and cur_state) or forceOneMode == 2:  # stuck
                dist_from_center_tether = random.default_rng().normal(loc=0.0, scale=np.sqrt(A), size=N_steps)
                angles = random.default_rng().uniform(0, 2 * np.pi, N_steps)
                x = prev_position[0] + np.cos(angles) * dist_from_center_tether
                y = prev_position[1] + np.sin(angles) * dist_from_center_tether
            else:  # cur_state is 0 or 1
                pass
            r[ind, 0] = x
            r[ind, 1] = y
            if forceOneMode==0:
                state_array[ind] = cur_state
            elif forceOneMode==1:
                state_array[ind] = 0
            elif forceOneMode==2:
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

    if saveFiles:
        # save now
        filepath = "./Trajectories/RonW/Salinity_" + str(salinity_index)
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        df.to_csv(path_or_buf=filepath + "/experimentCSV_" + str(len(os.listdir(filepath))) + ".csv")

    if experiment_name != []:
        df["experiment"] = experiment_name
    else:
        df["experiment"] = "synthetic"
    df = postProcessing(df)

    return df
