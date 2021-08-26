# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importData import importAll
from importData import removeImmobileParticle
import displayData
import cProfile
import pstats
import io
from scipy import stats
from extras import profile

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = importAll()
    df = df[(df["experiment"]=="G on G ; s=160") |
            (df["experiment"]=="K on K ; s=160"  )]
    for i,exp in enumerate(df.experiment.unique()):
        d = df.groupby("experiment").get_group(exp).groupby("particle").apply(len).to_frame()
        d = d.rename(columns={0: "len"})
        d.loc[:,"experiment"]=exp
        if i==0:
            D=d
        else:
            D = D.append(d)
    sns.histplot(data=D,x="len",hue="experiment")
    ax = plt.gca()
    ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.show()
    #
    # print(trajLengths.std())
    # print((trajLengths>300).sum())
    # print(len(trajLengths))
    # h = np.histogram(trajLengths,bins=1000)
    # XX = h[1]
    # Y = h[0]
    # X = (XX[range(1,len(XX))]+XX[range(0,len(XX)-1)])/2
    # # with open("stamX", 'w+') as f:
    # #     f.write(str(X))
    # # with open("stamY", 'w+') as f:
    # #     f.write(str(Y))
    # plt.hist(trajLengths,bins=1000)
    # plt.plot(X,Y)
    # plt.show()

    #displayData.tempHistGK(df)
    # df = df.groupby("experiment").get_group("G on G ; s=160")
    # X = df.dx.dropna().to_numpy()
    # H=np.histogram(X,bins=5000)
    # XX = H[1]
    # Y = H[0]
    # XX = (XX[range(1,len(XX))]+XX[range(0,len(XX)-1)])/2
    # ax = plt.gca()
    # plt.plot(XX,Y)
    # df.dx.hist(bins=5000,ax=ax)
    # plt.show()


