import numpy as np
# from autograd import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importData import importAll
import displayData
import bayesianTools
from generateTrajectories import generateDiffuseTetherModel
import extras

if __name__ == '__main__':
    df,padded_df = importAll()
    #Do some functions from here...