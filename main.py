import numpy as np
# from autograd import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importData import importAll
import displayData
import bayesianTools
from generateTrajectories import generateDiffuseTetherModel
import extras
import importData

if __name__ == '__main__':
    df = importAll(getLatest=False, isPar=True,breakTrajectoryAtCollisions=True, assignTrajStates=True)
    # dt = 1 / 30
    # T_end = (2 ** 10 - 1) * dt
    # N_particle = 1000
    # undersample_ratio = 0  # 0.1
    # saveFiles =    False
    # init_state = 0
    # forceOneMode = 0
    # T_stick = 100 * dt * 2 * 2
    # T_unstick = 40 * dt * 2 * 2
    # D = 0.3
        # A = 0.1 ** 2
    # isPar = True
    # df = generateDiffuseTetherModel(T_stick=T_stick, T_unstick=T_unstick, D=D, A=A, dt=dt, T_end=T_end,
    #                                 N_particle=N_particle, init_state=init_state, undersample_ratio=undersample_ratio,
    #                                 saveFiles=saveFiles, forceOneMode=forceOneMode, isPar=isPar)
    # from config import TRAJECTORIES_DIR, DATA_DIR, IMAGING_PARAMETERS_PATH, MODEL_PARAMETERS_PATH, \
    #     PADDING_INTERPOLATION_MAX_FRAMES
    # from importData import postProcessing
    # df_path = DATA_DIR + "/df" + "_synth1_ground_truth.fea"
    # df.to_feather(df_path)
    #
    # df = postProcessing(df,assignTrajStates=True,breakTrajectoryAtCollisions=False,undersampleLongTrajectories=False,isPar=True)
    # df_path = DATA_DIR + "/df" + "_synth1_assigned_states.fea"
    # df.to_feather(df_path)
    #
    #
    # dt = 1 / 30
    # T_end = (2 ** 10 - 1) * dt
    # N_particle = 1000
    # undersample_ratio = 0  # 0.1
    # saveFiles = False
    # init_state = 0
    # forceOneMode = 0
    # T_stick = 100 * dt * 2 * 2
    # T_unstick = 70 * dt * 2 * 2
    # D = 0.3
    # A = 0.01 ** 2
    # isPar = True
    # df = generateDiffuseTetherModel(T_stick=T_stick, T_unstick=T_unstick, D=D, A=A, dt=dt, T_end=T_end,
    #                                 N_particle=N_particle, init_state=init_state, undersample_ratio=undersample_ratio,
    #                                 saveFiles=saveFiles, forceOneMode=forceOneMode, isPar=isPar)
    # from config import TRAJECTORIES_DIR, DATA_DIR, IMAGING_PARAMETERS_PATH, MODEL_PARAMETERS_PATH, \
    #     PADDING_INTERPOLATION_MAX_FRAMES
    # from importData import postProcessing
    # df_path = DATA_DIR + "/df" + "_synth2_ground_truth.fea"
    # df.to_feather(df_path)
    #
    # df = postProcessing(df,assignTrajStates=True,breakTrajectoryAtCollisions=False,undersampleLongTrajectories=False,isPar=True)
    # df_path = DATA_DIR + "/df" + "_synth2_assigned_states.fea"
    # df.to_feather(df_path)

