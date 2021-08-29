# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importData import importAll
import displayData
import cProfile
import pstats
import io
import scipy
from scipy import stats
from extras import profile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = importAll()

    df = df[(df["experiment"] == "G on G ; s=160") |
           (df["experiment"] == "G on G ; s=60") |
           # (df["experiment"] == "G on G ; s=0") |
           (df["experiment"] == "K on K ; s=0") |
           (df["experiment"] == "R on W ; s=0")]


    displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(1, 100 + 1, step=1),
                                          preStr="combined1_", video_name="combined1", gif_name="combined1",
                                          var_name="combined1", xlim=60)
    displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(2, 100 + 1, step=2),
                                          preStr="combined2_", video_name="combined2", gif_name="combined2",
                                          var_name="combined2", xlim=60)
    displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(4, 100 + 1, step=4),
                                          preStr="combined4_", video_name="combined4", gif_name="combined4",
                                          var_name="combined4", xlim=60)
    displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(8, 100 + 1, step=8),
                                          preStr="combined8_", video_name="combined8", gif_name="combined8",
                                          var_name="combined8", xlim=60)
