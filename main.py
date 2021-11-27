from importData import importAll
import displayData
import bayesianTools
from generateTrajectories import generateDiffuseTetherModel
import extras
import numpy as np
import matplotlib.pyplot as plt
# from autograd import numpy as np
import os
import scipy
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

if __name__ == '__main__':
    df = importAll(isPar=True)

    #
    # plt.show(block=True)
    #
    # df = df[(df.experiment == "K on K ; s=0") |
    #         (df.experiment == "K on K ; s=60") |
    #         (df.experiment == "K on K ; s=110") |
    #         (df.experiment == "K on K ; s=160") |
    #         (df.experiment == "K on G ; s=160")]
    # # displayData.plot_MSD(df, lagtime=np.arange(1, 101), saveFiles=True, logscale=False)
    # # plt.show(block=True)
    # displayData.animate_G_dx_dt(df,d_array=np.arange(1,151))
    # plt.show(block=True)
    #
    # dt = 0.02
    # T_end = 0.02 * (128 - 1)
    # N_particle = 1
    # undersample_ratio = 0.0
    # saveFiles = False
    # init_state = 0
    # forceOneMode = 0
    # T_stick = .4
    # T_unstick = 1.
    # D = 60
    # A = .1
    # df = generateDiffuseTetherModel(T_stick=T_stick, T_unstick=T_unstick, D=D, A=A, dt=dt, T_end=T_end,
    #                                 N_particle=N_particle, init_state=init_state, undersample_ratio=undersample_ratio,
    #                                 experiment_name=[], salinity_index=1, saveFiles=saveFiles,
    #                                 forceOneMode=forceOneMode)
    # # df["state"]=0
    # # displayData.quickTraj(df,N_particles=1)
    #
    # trajData = bayesianTools.extract_trajData_from_df(df)
    # ind = df["state"] == 0.5
    # ind[0] = True
    # t_stick_unstick_with_boundary = df.t[ind]
    # t_stick_unstick = t_stick_unstick_with_boundary[1:-1]
    # print(t_stick_unstick)
    # dt_for_jac = 0.02 * np.ones(len(t_stick_unstick))
    # log_P_k = np.log(bayesianTools.kEventsProb(len(t_stick_unstick), 0, df.t.values.max(), T_stick=T_stick,
    #                                            T_unstick=T_unstick).astype(
    #     float) * np.abs(1))  # involve init state prior
    # true_likelihood = log_P_k + bayesianTools.specificTrajectoryLikelihood(trajData, 0, t_stick_unstick,
    #                                                                        dt_for_jac, T_stick=T_stick,
    #                                                                        T_unstick=T_unstick,
    #                                                                        D=D, A=A)
    #
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # ax = plt.gca()
    # displayData.plotTrajectories(df, t_end=2000, startAtOrigin=True, doneTrajNoMarker=True, ax=ax)
    # ax.set_title("True ; L=" + str(true_likelihood))
    # plt.show(block=False)
    # # t_st = time.time()
    # # p=extras.profile("on",[])
    # N = len(df)
    # half_N = int(N / 2)
    # df1, max_likelihood1, P_k1 = bayesianTools.particleLikelihood(df.head(half_N), T_stick=T_stick, T_unstick=T_unstick,
    #                                                               D=D, A=A,
    #                                                               mode=1)
    # df2, max_likelihood2, P_k2 = bayesianTools.particleLikelihood(df.tail(half_N), T_stick=T_stick, T_unstick=T_unstick,
    #                                                               D=D, A=A,
    #                                                               mode=1)
    # df_unsplit, max_likelihood, P_k = bayesianTools.particleLikelihood(df, T_stick=T_stick, T_unstick=T_unstick, D=D,
    #                                                                    A=A,
    #                                                                    mode=1)
    # df_split = df1.append(df2)
    #
    # plt.subplot(1, 3, 2)
    # ax = plt.gca()
    # displayData.plotTrajectories(df_unsplit, t_end=2000, startAtOrigin=True, doneTrajNoMarker=True, ax=ax)
    # ax.set_title("Unsplit ; L=" + str(max_likelihood))
    # plt.subplot(1, 3, 3)
    # ax = plt.gca()
    # displayData.plotTrajectories(df_split, t_end=2000, startAtOrigin=True, doneTrajNoMarker=True, ax=ax)
    # ax.set_title("Split ; L=" + str(max_likelihood1 + max_likelihood2 + P_k - P_k1 - P_k2))
    # plt.show(block=False)
    #
    # plt.figure()
    # plt.plot(df.state)
    # plt.plot(df_unsplit.state, linestyle="--")
    # plt.plot(df_split.state, linestyle=":")
    # plt.show(block=False)
    #
    # print("")
    #
    # plt.show(block=True)

# df = bayesianTools.assignStateBayesian(df, T_stick=1., xT_unstick=1.3, D=60, A=.1)
# t_end = time.time()
# print(str((t_end-t_st)*1000) + "ms")
# extras.profile("off", p)
#
# displayData.plotTrajectories(df_orig, t_end=2000, startAtOrigin=True, doneTrajNoMarker=True)
# ax = plt.gca()
# ax.set_title = ("True")
# displayData.plotTrajectories(df, t_end=2000, startAtOrigin=True, doneTrajNoMarker=True)
# ax = plt.gca()
# ax.set_title = ("Calculated")
# plt.show()

# x = np.linspace(0, 3 * np.pi, 500)
# y = np.sin(x)
# dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
#
# # Create a set of line segments so that we can color them individually
# # This creates the points as a N x 1 x 2 array so that we can stack points
# # together easily to get the segments. The segments array for line collection
# # needs to be (numlines) x (points per line) x 2 (for x and y)
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
# fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
# plt.show(block=False)
# # # Create a continuous norm to map from data points to colors
# # norm = plt.Normalize(dydx.min(), dydx.max())
# # lc = LineCollection(segments, cmap='viridis', norm=norm)
# # # Set the values used for colormapping
# # lc.set_array(dydx)
# # lc.set_linewidth(2)
# # line = axs[0].add_collection(lc)
# # fig.colorbar(line, ax=axs[0])
#
# # Use a boundary norm instead
# cmap = ListedColormap(['r', 'g', 'b'])
# norm = BoundaryNorm([-1, 0, 1], cmap.N)
# lc = LineCollection(segments, cmap=cmap, norm=norm)
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs[1].add_collection(lc)
# fig.colorbar(line, ax=axs[1])
#
# axs[0].set_xlim(x.min(), x.max())
# axs[0].set_ylim(-1.1, 1.1)
# plt.show()

# df = generateTrajectories.generateDiffuseTetherModel(T_stick=0.8, T_unstick=2.5, D=60., A=1., dt=0.02, T_end=6.0,
#                                                      N_particle=50,
#                                                      init_state=0,
#                                                      undersample_ratio=0.10, salinity_index=1, saveFiles=False,
#                                                      forceOneMode=0)
# bayesianTools.basicMaxLikelihood(df, x0=[5, 30], scaling_factor=1., usePar=True)
# print(df)
# df = df[#(df["experiment"] == "K on K ; s=0") |
#         (df["experiment"] == "G on G ; s=0") |
# (df["experiment"] == "G on G ; s=160") |
# # (df["experiment"] == "G on G ; s=60")]# |
# #(df["experiment"] == "R on W ; s=1")]
#
# #displayData.animate_G_dx_dt(df,d_array=[10],saveFiles=False,showFigures=True)
# displayData.plot_MSD(df,lagtime=np.arange(1,21,step=1),logscale=False)
# plt.show()

# df = df.head(800)
# df = generateTrajectories.generateDiffuseTetherModel(T_stick=1.99, T_unstick=1., D=60., A=1., dt=0.02, T_end=4.0,
#                                                      N_particle=10,
#                                                      init_state=0,
#                                                      undersample_ratio=0.0, salinity_index=1, saveFiles=False,
#                                                      forceOneMode=0)
#
# displayData.animate_Trajectories(df,t_end_array=np.arange(0,101),showFigure=True,saveFiles=False)
# plt.show()
# # bayesianTools.basicMaxLikelihood(df, x0=[6,30], scaling_factor=1.,usePar=False)
#
# L = lambda A: -1 * bayesianTools.experimentLikelihood(df, 0.1, 100., 1, A)
# cons = ({'type': 'ineq',
#          'fun': lambda x: x - 1.})
# minimum = scipy.optimize.minimize(L, x0=10., constraints=cons)
# print(minimum)

# bayesianTools.basicMaxLikelihood(df, x0=np.array([20, 0.15, 20, 14.]), usePar=True)
# D_array = np.linspace(0.05, 6., num=20)
# L_array = 0*D_array
# for i,D in enumerate(D_array):
#     L_array[i] = L(D)
# plt.plot(D_array,L_array)
# plt.show()

# df = importAll()
# df = df[(df["experiment"] == "K on K ; s=0") |
#         (df["experiment"] == "G on G ; s=0")]
#
# dx, _, _ = displayData.getDiffs(df[(df["experiment"] == "G on G ; s=0")], dt=3)
# print(np.std(dx))
# dx, _, _ = displayData.getDiffs(df[(df["experiment"] == "K on K ; s=0")], dt=3)
# print(np.std(dx))
#
# kp = displayData.plot_G_dx_dt(df, dt=3, gridsize=2000, direction="x")
#
# ax = plt.gca()
# L = ax.get_lines()
#
#
# def gauss(xi, a, x0, sigma):
#     return a * np.exp(-(xi - x0) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
#
#
# for i in range(2):
#     x = L[i].get_xdata()
#     y = L[i].get_ydata()
#     popt, pcov = scipy.optimize.curve_fit(gauss, x, y, p0=[1., 0., 3.])
#     center = popt[1]
#     sigma = popt[2]
#     plt.plot(x, gauss(x, 1., center, sigma), linestyle="--", color=L[i].get_color())
#     print(center)
#     print(sigma)
#     print(np.sqrt(np.trapz((x ** 2) * y, x)))
#     print("yo")
#
#     ax.set_ylim([1e-4, 1])
#     # ax.set_yscale("linear")
#     # print(np.std(dx))
# plt.show()

# df = df[(df["experiment"] == "R on W ; s=1")]
# df = df.head(85 * 50 - 10)
# df = generateTrajectories.generateDiffuseTetherModel(T_stick=0.8, T_unstick=2.5, D=60., A=0.1, dt=0.02, T_end=3.0,
#                                                  N_particle=50,
#                                                  init_state=0,
#                                                  undersample_ratio=0.15, salinity_index=2, saveFiles=True,forceOneMode=1)

# displayData.animate_Trajectories(df,t_end_array=np.arange(0,151),saveFiles=False,showFigure=True)
# bayesianTools.basicMaxLikelihood(df, x0=np.array([1.2, 1.9, 40, 1]),usePar=True)

# df = importAll()
# df = df[(df["experiment"] == "R on W ; s=1")]
# # displayData.animate_Trajectories(df,saveFiles=False,showFigure=True,t_end_array=np.arange(100))
# displayData.animate_G_dx_dt(df,saveFiles=True,showFigures=False)
# plt.show()
# df = importAll()
# N_particles = 10
# df = df[(df["experiment"] == "K on K ; s=0")]
# df = df.head(1000)
# bayesianTools.experimentLikelihood(df, 1, 2, 1, 1, False)

# df = df[(df["experiment"] == "K on K ; s=0") |
#         (df["experiment"] == "G on G ; s=0")]
# displayData.animate_G_dx_dt(df, d_array=[3],xlim=15,ylim=[1e-4,0.8],x="x")
# displayData.plot_MSD(df, lagtime=np.arange(1, 21, step=1), saveFiles=True, logscale=False, eqParticleWeight=False)
# displayData.plot_MSD(df,lagtime=np.arange(1,31,step=10),saveFiles=False,logscale=False,eqParticleWeight=True,ax=plt.gca())
# ax = plt.gca()
# ax.set_xlim([0,0.42])
# plt.show()
# displayData.quickTraj(df, N_particles=N_particles)
# random_particle_id = np.random.choice(df.particle.unique(), size=N_particles, replace=False)
#
# small_df = df[df["particle"].isin(random_particle_id)]
# dx, dy = displayData.getDiffs(df, 1)
# plt.hist(np.sort(np.abs(dx)))
# print(np.mean(np.sort(np.abs(dx))))
# plt.show()
# df = df[(df["experiment"] == "G on G ; s=160") |
#         (df["experiment"] == "G on G ; s=60") |
#         (df["experiment"] == "G on G ; s=0") |
#         (df["experiment"] == "K on K ; s=0")]
# displayData.animate_G_dx_dt(df,d_array=np.arange(1,201),saveFiles=True,showFigures=False)
# # plt.show()
# displayData.quickTraj(df,N_particles=6)
# gb = df.groupby("particle")
# random_particle_id = np.random.choice(df.particle.unique(), size=1, replace=False)[0]
# displayData.animate_Trajectories(gb.get_group(random_particle_id), showFigure=True, t_end_array=np.arange(0, 101),saveFiles=False)
# plt.show()

# bayesianTools.basicMaxLikelihood(df[(df["experiment"] == "K on K ; s=0")], x0=[1.5, 2.8, 25., 0.1], usePar=True,
#                                  maxIter=100, savefile_extra="KK0")
# bayesianTools.basicMaxLikelihood(df[(df["experiment"] == "G on G ; s=160")], x0=[1.5, 2.8, 25., 0.1], usePar=True,
#                                  maxIter=100, savefile_extra="GG160")
# bayesianTools.basicMaxLikelihood(df[(df["experiment"] == "RW")], x0=[1.5, 2.8, 25., 0.1], usePar=True,
#                                  maxIter=100, savefile_extra="RW")
# bayesianTools.basicMaxLikelihood(df[(df["experiment"] == "G on G ; s=60")], x0=[1.5, 2.8, 25., 0.1], usePar=True,
#                                  maxIter=100, savefile_extra="GG60")
# bayesianTools.basicMaxLikelihood(df[(df["experiment"] == "G on G ; s=0")], x0=[1.5, 2.8, 25., 0.1], usePar=True,
#                                  maxIter=100, savefile_extra="GG0")

# df = df[(df["experiment"] == "G on G ; s=160") |
#         (df["experiment"] == "G on G ; s=60") |
#         (df["experiment"] == "G on G ; s=0") |
#         (df["experiment"] == "K on K ; s=0")]
# MSD_df = displayData.plot_MSD(df,lagtime=np.arange(1, 30, step=1), marker="*", logscale=True)
# np.save("tempthing",[MSD_df],datatype="object")

# plt.show()
# x = np.arange(5, 8)
# y = np.arange(17, 21)
# for i,j in np.nditer([x,y],flags=['multi_index']):
#     print(str(x[i])+","+str(y[j]))

# df = importAll(300)
# # df = df[(df["experiment"] == "G on G ; s=160")]
# df = df[(df["experiment"] == "Lab")]
# df = df.head(8000)
# # x0 = [0.5, 0.8, 50., 5]
#
#
# T_stick_arr = np.linspace(5, 5, 1)
# T_unstick_arr = np.linspace(0.1, 5, 1)
# D_arr = np.linspace(5., 60., 51)
# A_arr = np.linspace(0.05, 5., 1)
#
# L_tensor = bayesianTools.gridLikelihood(df, T_stick_arr, T_unstick_arr, D_arr, A_arr)
# plt.plot(D_arr,L_tensor)

#
# plt.gca()
# plt.show()

# D_arr = np.exp(np.linspace(-5., 6., 10))
# A_arr = np.exp(np.linspace(-5., 3., 10))
# D_arr = np.linspace(20, 50, 1)
# A_arr = np.linspace(0.05, 0.5, 1)
# T_stick_arr = np.linspace(1, 5, 1)
# T_unstick_arr = np.linspace(1, 5, 1)

# plt.plot(A_arr, L_tensor)
# plt.show()

# df = df[(df["experiment"] == "G on G ; s=160") |
#        (df["experiment"] == "G on G ; s=60") |
#         (df["experiment"] == "G on G ; s=0") |
#        (df["experiment"] == "K on K ; s=0")]

# df = df[df["experiment"] == "G on G ; s=60"]
# df = pd.concat([g[1] for g in list(df.groupby("particle"))[:70]])
# particles = df.particle.unique()
# gb = df.groupby("particle")
# for i, particle_id in enumerate(particles):
#     D = gb.get_group(particle_id)
#     x0 = D['x'].head(1).to_numpy()[0]
#     y0 = D['y'].head(1).to_numpy()[0]
#     D['x'] = D['x'].add((np.random.uniform()-0.5)*100 - x0)
#     D['y'] = D['y'].add((np.random.uniform()-0.5)*100 - y0)
#     if i == 0:
#         dff = D
#     else:
#         dff = dff.append(D)
#
#     df = dff

# ax=plt.gca()
# p=displayData.plotTrajectories(df, groupby="particle", t_st=0, t_end=1e5, doneTrajNoMarker=True,
#                 startAtOrigin=True,ax=ax)
# plt.show()
# print("Done!")
# plt.figure()
# sns.kdeplot(data=df, x="dx", hue="particle", common_norm=False)
# plt.show()

#
# df = df[(df["experiment"] == "G on G ; s=160") |
#        (df["experiment"] == "G on G ; s=60") |
#        # (df["experiment"] == "G on G ; s=0") |
#        (df["experiment"] == "K on K ; s=0") |
#        (df["experiment"] == "RW")]
#
#
# displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(1, 100 + 1, step=1),
#                                       preStr="combined1_", video_name="combined1", gif_name="combined1",
#                                       var_name="combined1", xlim=60)
# displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(2, 100 + 1, step=2),
#                                       preStr="combined2_", video_name="combined2", gif_name="combined2",
#                                       var_name="combined2", xlim=60)
# displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(4, 100 + 1, step=4),
#                                       preStr="combined4_", video_name="combined4", gif_name="combined4",
#                                       var_name="combined4", xlim=60)
# displayData.animate_G_dx_dt_with_conv(df, d_array=np.arange(8, 100 + 1, step=8),
#                                       preStr="combined8_", video_name="combined8", gif_name="combined8",
#                                       var_name="combined8", xlim=60)
