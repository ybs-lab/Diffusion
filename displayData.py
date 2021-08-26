import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import cv2
import argparse
from datetime import datetime


def plot_G_dx_dt(df, direction="x", groupby="experiment", dt=1, kind="kde", filter_diff=0, return_stats=False,
                 **kwargs):
    # Plots G(dx,dt)
    # Inputs: 1. dataframe with trajectories
    #         2. parameter to group by (default is particle)
    #         3. dt - time interval in frame units
    #         4. kind = "kde" or "hist
    #         5. filter_diff - throw away small dx/dy (|dx|<filter_diff)), useful to study just the tails
    #         6. return_stats - option for the function to return a small statistics dataframe
    #         7. keyword arguments to pass to dataFrame.plot() function (don't include kind here)
    #         Note: if axes is not specified, a new one is created
    #               if marker is not specified, . is selected.
    # Outputs:
    #         This function returns:
    #         1. returns handle to the kde/histogram plots
    #         2. if return_stats==True, also return stats dataframe
    #         In addition this function does:
    #         1. plots trajectories on the input axes / new one and draws a figure
    #         2. makes the above axes with equal scale

    if "ax" not in kwargs:
        fig = plt.figure()
        ax = plt.axes()
        kwargs["ax"] = ax
    else:
        ax = kwargs["ax"]

    ax.set_xlabel("\u0394" + direction + "[\u03BCm]")
    ax.set_ylabel("G")
    ax.set_title("G(\u0394" + direction + ",\u0394t=" + str(dt) + " frames)")
    ax.set_yscale('log')

    dx = "d" + direction  # this can be dx or dy
    D = df.copy()  # to stop those pesky warnings
    D = D[set(["particle", "frame", direction, groupby])]
    D = D.astype({'frame': 'float'})
    D_offset = D.copy()
    fraction = 0.5
    D_offset.loc[:, "frame"] = D_offset.loc[:, "frame"].add(dt).add(
        fraction)  # add fraction for sorting later because we want D_offset - D
    D = D.append(D_offset).sort_values(by=['particle', 'frame'])
    G = D[["particle", "frame", direction]].groupby("particle").diff()
    # G["particle"]=D["particle"]
    G[groupby] = D[groupby]
    G = G[G["frame"] == fraction]  # this makes us take only the dt we wanted
    G = G.rename(columns={direction: dx})

    G = G[G[dx].abs() > filter_diff]

    # we need this if instead of using sns.displot because we need to specify the ax input (in kwargs)
    if kind == "kde":
        kp = sns.kdeplot(data=G, x=dx, hue=groupby, **kwargs,
                         palette=sns.color_palette("hls", len(G.groupby(groupby))))
    elif kind == "hist":
        kp = sns.histplot(data=G, x=dx, hue=groupby, **kwargs,
                          palette=sns.color_palette("hls", len(G.groupby(groupby))))
    else:
        return

    if not return_stats:
        return kp
    else:
        # get stats for dx, not related to plot
        grouped_df = G.groupby(groupby)
        stats_df = grouped_df.count().rename(columns={dx: "count"})
        stats_df.loc[:, "dt"] = dt
        stats_df.loc[:, "mean"] = grouped_df.mean().loc[:, dx]
        stats_df.loc[:, "var"] = grouped_df.var().loc[:, dx]
        return kp, stats_df


def plotTrajectories(df, groupby="particle", t_st=0, t_end=1e5, doneTrajNoMarker=False,
                     startAtOrigin=False, **kwargs):
    # Plots the trajectories of all the particles in the dataframe
    # NOTE: on my PC this gets pretty slow with more than 200 particles
    #       so filter the dataframe before calling this (Todo: improve plotting to allow many trajectories)
    # Inputs: 1. dataframe with trajectories
    #         2. parameter to group by (default is particle)
    #         3. t_start in frame units (default is 0 so plot from the start)
    #         4. t_end in frame units (default is t_end>all traj lengths so just plot until the end)
    #         5. doneTrajNoMarker - remove markers from trajectories with length < t_end
    #         6. startAtOrigin - all particles begin movement at x=0, y=0
    #         7. keyword arguments to pass to dataFrame.plot() function
    #         Note: if axes is not specified, a new one is created
    #               if marker is not specified, x is selected.
    # Outputs:
    #         1. returns list of lines (plot handles)
    #         2. plots trajectories on the input axes / new one and draws a figure
    #         3. makes the above axes with equal scale

    # If axes is not specified, create a new one
    if "ax" not in kwargs:
        fig = plt.figure()
        ax = plt.axes()
        kwargs["ax"] = ax
    else:
        ax = kwargs["ax"]

    if "legend" in kwargs:  # to force legend=off regardless of kwargs
        kwargs.pop("legend")

    if "marker" not in kwargs:
        kwargs["marker"] = "x"

    df = df.copy()  # we want to filter and not affect the input df

    if doneTrajNoMarker:
        df["t_end"] = df.groupby('particle')['frame'].transform(
            'last')  # mark end time for each particle trajectory

    df = df[(df.frame >= t_st) & (df.frame <= t_end)]  # filter by time
    if startAtOrigin:  # for each particle, subtract x[0] and y[0] so the traj starts at the origin
        df['x'] = df['x'] - df.groupby('particle')['x'].transform('first')
        df['y'] = df['y'] - df.groupby('particle')['y'].transform('first')

    p = df.groupby(groupby).plot(x="x", y="y", legend=False, **kwargs)
    p = p[p.index[0]].get_lines()

    ax.axis('equal')
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    if doneTrajNoMarker:
        # get index of plots to remove markers from according to particles with trajs that dont last after t_end
        ind = np.where(np.isin(df.particle.unique(), df.particle[df.t_end <= t_end].unique()))[
            0]  # what plot indices to remove marker
        for i in ind:
            p[i].set_marker('None')
    return p


def createGif(path='.', buffer=5, output_name="mygif", image_type="png"):
    # createGif creates a gif file in the local folder from images in a folder according to their alphabetical order
    # IMPORTANT NOTE 1: images names should have preceding zeros, e.g. 10 follows 1 and precedes 2 so use 01, 02,...,10
    # Note 2: if a gif with the existing name exists, it will be copied with a time suffix
    #
    # Inputs: 1. path to images
    #         2. buffer (extra frames at the end of gif so it won't look weird when restarting)
    #         3. name of output gif
    #         4. image type to parse (default is png)
    #
    # Outputs: 1. gif file in local folder

    images = []
    for f in os.listdir(path):
        if f.endswith(image_type):
            images.append(f)

    output_full = output_name + '.gif'
    backupBeforeSave(output_full)

    with imageio.get_writer(output_name + '.gif', mode='I') as writer:
        for i, filename in enumerate(images):
            image = imageio.imread(path + "\\" + filename)
            writer.append_data(image)
            # replicate the last frame a few more times (buffer)
            if i == (len(images) - 1):
                for n in range(0, buffer):
                    writer.append_data(image)

    output_string = "The output gif is {}".format(output_full)
    print(output_string)
    return


def createVideo(path='.', fps=4.0, output_name="myvideo", image_type="png"):
    # createVideo creates an mp4 file in the local folder from images in a folder according to their alphabetical order
    # IMPORTANT NOTE 1: images names should have preceding zeros, e.g. 10 follows 1 and precedes 2 so use 01, 02,...,10
    # Note 2: if a video with the existing name exists, it will be copied with a time suffix
    # Note 3: this usually doesn't work when called from console
    #
    # Inputs: 1. path to images
    #         2. fps
    #         3. name of output video
    #         4. image type to parse (default is png)
    #
    # Outputs: 1. mp4 file in local folder
    #
    # Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default=image_type, help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, default=(output_name + '.mp4'), help="output video file")
    args = vars(ap.parse_args())
    # Arguments
    ext = args['extension']
    output = args['output']

    images = []
    for f in os.listdir(path):
        if f.endswith(ext):
            images.append(f)

    # Determine the width and height from the first image
    image_path = os.path.join(path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    backupBeforeSave(output)

    # This creates a new file
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    for image in images:

        image_path = os.path.join(path, image)
        frame = cv2.imread(image_path)

        out.write(frame)  # Write out frame to video

        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    output_string = "The output video is {}".format(output)
    print(output_string)
    return


def backupBeforeSave(filepath):
    # If a file exists, back it up by renaming it with the current date and time
    # Input: 1. filepath = full path to file, including suffix of file
    # Output: Renames the file if it exists (else does nothing)
    if os.path.isfile(filepath):  # if a file exists with the same name, rename existing file
        os.rename(filepath, filepath[0:filepath.find('.')] + datetime.now().strftime('__%d_%m__%H_%M_%S') + filepath[
                                                                                                            filepath.find(
                                                                                                                '.'): len(
                                                                                                                filepath)])


def animate_G_dx_dt(df, d_array=np.arange(1, 101), x="x", kind="kde", video_name="myvideo", gif_name="mygif",
                    var_name="myvar"):
    # Create gif and mp4 and variance plot of G(delta x/y, delta t)
    # Inputs:  1. dataframe
    #          2. d_array - delta t array in frame units
    #          3. x = "x" or "y"
    #          4. mode = "kde" or "hist"
    # Outputs: 1. mp4 file with G(dx,dt) animation
    #          2. gif file with G(dx,dt) animation
    #          3. png file variance and counts vs dt (var is solid line, counts dashed)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel(fr'$\Delta {x} [\mu m]$')
    ax.set_ylabel("G")
    # fig.canvas.draw()
    # plt.show(block=False)
    filenames = []
    files_path = ".\\images_to_vid\\"
    for i, d in enumerate(d_array):
        plt.cla()
        ax.set_xlim([-15, 15])
        if kind == "kde":
            ax.set_ylim([1e-6, 1])
        ax.set_yscale('log')
        kp, cur_stats_df = plot_G_dx_dt(df, dt=d, ax=ax, direction=x, groupby="experiment", kind=kind,
                                        filter_diff=0.0, return_stats=True)
        ax.set_title(fr'$G(\Delta {x},\Delta t= {d}$' + " frames)")
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # create file name and append it to a list
        filename = files_path + f'{i:04d}.png'
        filenames.append(filename)
        # save frame
        plt.savefig(filename, format="png", dpi=400)
        # handle the stats df
        if i == 0:
            stats_df = cur_stats_df
        else:
            stats_df = stats_df.append(cur_stats_df)

    createVideo(path=files_path, fps=6.0, output_name=video_name, image_type="png")
    createGif(path=files_path, buffer=5, output_name=gif_name, image_type="png")

    for f in os.listdir(files_path):
        os.remove(files_path + f)

    if len(stats_df) > 0:
        backupBeforeSave(os.getcwd() + "\\stats_df.fea")
        stats_df.reset_index().to_feather(os.getcwd() + "\\stats_df.fea")
        stats_df = pd.read_feather(os.getcwd() + "\\stats_df.fea")
        stats_fig = plt.figure()
        sns.lineplot(data=stats_df, x="dt", y="var", hue="experiment")
        ax1 = plt.gca()
        ax1.set_xlabel(r'$\Delta t [frames]$')
        ax2 = plt.twinx()
        sns.lineplot(data=stats_df, x="dt", y="count", hue="experiment", linestyle=":", ax=ax2)
        ax2.get_legend().remove()

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

        ax1.set_ylabel(rf'$Var(G(\Delta {x},\Delta t)) [\mu m^2]$')

        backupBeforeSave(os.getcwd() + "\\" + var_name + ".png")
        plt.savefig(var_name + ".png", format="png", dpi=400)
    return


def animate_Trajectories(df, t_end_array=np.arange(0, 501), startAtOrigin=True, fixedLimits=True,
                         video_name="traj_video", gif_name="traj_gif"):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel(r'$x-x_{0} [\mu m]$')
    ax.set_xlabel(r'$y-y_{0} [\mu m]$')

    if startAtOrigin and fixedLimits:
        lim = max(
            (df['x'] - df.groupby('particle')['x'].transform('first')).abs().max(),
            (df['y'] - df.groupby('particle')['y'].transform('first')).abs().max())

    # fig.canvas.draw()
    # plt.show(block=False)
    filenames = []
    files_path = ".\\images_to_vid\\"
    for i, t_end in enumerate(t_end_array):
        plt.cla()
        p = plotTrajectories(df, t_end=t_end, doneTrajNoMarker=True, startAtOrigin=startAtOrigin, ax=ax)
        ax.set_title(f't = {t_end} frames')
        if fixedLimits:  # keep the same limits each loop iteration
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])

        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # create file name and append it to a list
        filename = files_path + f'{i:04d}.png'
        filenames.append(filename)
        # save frame
        plt.savefig(filename, format="png", dpi=400)
        # handle the stats df

    createVideo(path=files_path, fps=20.0, output_name=video_name, image_type="png")
    createGif(path=files_path, buffer=5, output_name=gif_name, image_type="png")

    for f in os.listdir(files_path):
        os.remove(files_path + f)


def tempHistGK(df,n_particle=1,period = 5):
    df_G = df.groupby("experiment").get_group("G on G ; s=160")
    df_K = df.groupby("experiment").get_group("K on K ; s=0")
    # df = pd.concat([df.groupby("experiment").get_group(exp) for i, exp in enumerate(["G on G ; s=160",

    for n in range(20):
        particles_G = np.random.choice(df_G.particle.unique(), n_particle)
        D_G = pd.concat([df_G.groupby("particle").get_group(particle) for i, particle in enumerate(particles_G)])
        #        plot_G_dx_dt(D_G, groupby="particle", legend=False, dt=(1 + n % 5),kind="hist",color="g",ax=ax)
        particles_K = np.random.choice(df_K.particle.unique(), n_particle)
        D_K = pd.concat([df_K.groupby("particle").get_group(particle) for i, particle in enumerate(particles_K)])
        #        plot_G_dx_dt(D_K, groupby="particle", legend=False, dt=(1 + n % 5),kind="hist",color="b",ax=ax)
        plot_G_dx_dt(D_G.append(D_K), groupby="experiment", legend=True, dt=(1 + n % period), kind="hist")

        ax = plt.gca()
        ax.set_xlim([-12, 12])
        # ax.set_ylim([1e-6, 1])
        ax.set_yscale('linear')
        plt.show()

    def trajLengthStats(df):
        for i, exp in enumerate(df.experiment.unique()):
            d = df.groupby("experiment").get_group(exp).groupby("particle").apply(len).to_frame()
            d = d.rename(columns={0: "len"})
            d.loc[:, "experiment"] = exp
            if i == 0:
                D = d
            else:
                D = D.append(d)
        sns.histplot(data=D, x="len", hue="experiment")
        fig = plt.figure()
        ax = plt.gca()
        ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.show()
