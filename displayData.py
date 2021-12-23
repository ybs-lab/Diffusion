import os

import cv2.cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import cv2
import argparse
import scipy.spatial
import scipy.optimize

import extras
from extras import backupBeforeSave
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

OUTPUTS_DIR = ".\\Outputs\\files"
IMG4VID_DIR = ".\\Outputs\\img4vid"


def getDiffs(df, dt, t_column_name="frame", is_df_padded=False):
    if is_df_padded:  # if ALL frames are filled
        diffs_df = df[["x", "y", "particle", "trajDuration"]]
        dxdy_df = diffs_df.groupby("particle").diff(int(dt))
        keep_inds = ~np.isnan(dxdy_df["x"].values)
        dx = dxdy_df.loc[keep_inds, "x"].values
        dy = dxdy_df.loc[keep_inds, "y"].values
        particle = diffs_df.loc[keep_inds, "particle"].values
        trajDuration = diffs_df.loc[keep_inds, "trajDuration"].values

    else:
        particles = df.particle.unique()
        gb = df.groupby("particle")
        dx = []
        dy = []
        particle = []
        trajDuration = []
        for particle_id in particles:
            df_particle = gb.get_group(particle_id)
            t = df_particle[t_column_name].values
            x = df_particle.x.values
            y = df_particle.y.values
            curTrajDuration = len(df_particle)
            inds = np.vstack(
                np.where(np.triu(
                    scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(t[:, None], 'cityblock'))) == dt))
            particle = np.concatenate([particle, particle_id * np.ones(len(inds[0, :]), dtype=int)])
            trajDuration = np.concatenate([trajDuration, curTrajDuration * np.ones(len(inds[0, :]), dtype=int)])
            dx = np.concatenate([dx, (x[inds[1, :]] - x[inds[0, :]])])
            dy = np.concatenate([dy, (y[inds[1, :]] - y[inds[0, :]])])

    return [dx, dy, particle, trajDuration]


def plot_MSD(df, lagtime=np.arange(1, 1000, step=5), group_by="experiment", logscale=True, ax="New",
             saveFiles=False, filename="myMSD", eqParticleWeight=True, lagTimeInFrames=False, isdfPadded=False,
             **kwargs):
    group_keys = df[group_by].unique()
    gb = df.groupby(group_by)
    MSD_df = pd.DataFrame([])
    counter = 0

    for i, dt in enumerate(lagtime):
        diff_df = pd.DataFrame([])
        for j, key in enumerate(group_keys):
            particle_df = gb.get_group(key)  # could be experiment df as well
            fps = particle_df.head(1).fps.values[0]
            out = getDiffs(particle_df, dt, is_df_padded=isdfPadded)
            dx = out[0]
            dy = out[1]
            particle = out[2]
            dr2 = dx ** 2 + dy ** 2
            cur_df = pd.DataFrame(dr2, columns=["dr2"])
            cur_df["dt"] = dt
            cur_df["particle"] = particle
            cur_df["fps"] = fps
            cur_df[group_by] = key
            diff_df = diff_df.append(cur_df)
        if (group_by != "particle") and eqParticleWeight:
            agg_d = {c: 'mean' if c == 'dr2' else 'first' for c in diff_df.columns}
            diff_df = diff_df.groupby("particle").agg(agg_d)
            # diff_df = diff_df.groupby("particle").mean()
            diff_df["particle"] = diff_df.index
            diff_df.index = np.arange(len(diff_df))
        cur_MSD_df = diff_df.groupby(group_by).mean()  # now mean
        cur_MSD_df[group_by] = cur_MSD_df.index
        cur_MSD_df.index = np.arange(len(cur_MSD_df)) + counter
        counter += len(cur_MSD_df)
        MSD_df = MSD_df.append(cur_MSD_df)

    MSD_df = MSD_df.rename(columns={"dr2": "MSD"})
    # MSD_df.loc[MSD_df["experiment"] == "Lab", "dt"] = MSD_df.loc[MSD_df["experiment"] == "Lab", "dt"] * fps / 20. #fix for lab... make this generic later

    MSD_df["dt_frames"] = MSD_df["dt"]
    MSD_df["dt"] = MSD_df["dt"].div(MSD_df["fps"])  # convert frames to sec

    if ax == "New":
        fig = plt.figure()
        ax = plt.gca()

    if lagTimeInFrames:
        ax.set_xlabel("Lag Time (frames)")
    else:
        ax.set_xlabel("Lag Time (sec)")

    ax.set_ylabel(r'${\rm MSD} [\mu {\rm m}^{2}]$')
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    iters = MSD_df[group_by].unique()
    MSD_gb = MSD_df.groupby(group_by)
    if lagTimeInFrames:
        xaxis_name = "dt_frames"
    else:
        xaxis_name = "dt"
    # sns.lineplot(data=MSD_df, x=xaxis_name, y="MSD", hue=group_by,ax=ax, **kwargs)

    for group_id in iters:
        group = MSD_gb.get_group(group_id)
        plt.plot(group[xaxis_name], group["MSD"], label=[group_by + ": " + str(group_id)], marker="*",
                 markersize=3,
                 **kwargs)
    ax.legend()
    plt.setp(ax.get_legend().get_texts(), fontsize='7')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='8')  # for legend title
    plt.grid()

    if saveFiles:
        filepath = OUTPUTS_DIR + "\\" + filename + ".png"
        backupBeforeSave(filepath)
        plt.savefig(filepath, format="png", dpi=400)
    return MSD_df


def plot_G_dx_dt(df, direction="xy", group_by="experiment", dt=1, kind="kde", filter_diff=0,
                 return_stats=False,
                 semilogscale=True, isdfPadded=False, equal_particle_weight=True,
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

    ax.set_xlabel(fr'$\Delta {direction} [\mu m]$')
    ax.set_ylabel(r'$G$')
    ax.set_title(fr'$G(\Delta {direction}, \Delta t = {dt}$ frames)')
    if semilogscale:
        ax.set_yscale('log')
    group_keys = df[group_by].unique()
    gb = df.groupby(group_by)
    G = pd.DataFrame([])
    for j, key in enumerate(group_keys):
        out = getDiffs(gb.get_group(key), dt, is_df_padded=isdfPadded)
        dx = out[0]
        dy = out[1]
        # particle = out[2]
        invTrajDuration = np.power(out[3], -1.)
        # dr2 = dx ** 2 + dy ** 2
        if direction == "xy":
            cur_df = pd.DataFrame(
                {"dxy": np.hstack([dx, dy]), "invTrajDuration": np.hstack([invTrajDuration, invTrajDuration])})
        else:
            cur_df = pd.DataFrame({"dx": dx, "dy": dy, "invTrajDuration": invTrajDuration})
        cur_df["dt"] = dt
        cur_df[group_by] = key
        G = G.append(cur_df)

    G = G.reset_index()
    dx_name = "d" + direction  # this can be dx or dy
    # # we need this if instead of using sns.displot because we need to specify the ax input (in kwargs)
    # if (kind == "hist") or ((group_by == "particle") or ((group_by != "particle") and not equal_particle_weight)):

    if equal_particle_weight:
        G["weights"] = G["invTrajDuration"]  # equal particle weight (add weight of inverse traj length)
    else:
        G["weights"] = 1  # equal step weight

    if "palette" not in kwargs:
        palette = sns.color_palette("tab10", len(G.groupby(group_by)))
        kwargs["palette"] = palette

    if kind == "kde":
        kp = sns.kdeplot(data=G, x=dx_name, hue=group_by, weights="weights", common_norm=False, **kwargs)
    elif kind == "hist":
        kp = sns.histplot(data=G, x=dx_name, hue=group_by, weights="weights", **kwargs,
                          stat="density")
    # palette=sns.color_palette("None", len(G.groupby(groupby))))
    else:
        return
    sns.move_legend(ax, "upper left")
    plt.setp(ax.get_legend().get_texts(), fontsize='7')  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='8')  # for legend title
    ax.grid(True)

    cur_ylim = np.asarray(ax.get_ylim())
    if semilogscale:
        cur_ylim[0] = np.asarray([cur_ylim[0], 1e-6]).max()
    else:
        cur_ylim[0] = 0.
    ax.set_ylim(cur_ylim)

    if not return_stats:
        return kp
    else:
        # get stats for dx, not related to plot
        grouped_df = G.groupby(group_by)
        stats_df = grouped_df.count().rename(columns={dx_name: "count"})
        stats_df.loc[:, "dt"] = dt
        stats_df.loc[:, "mean"] = grouped_df.mean().loc[:, dx_name]
        stats_df.loc[:, "var"] = grouped_df.var().loc[:, dx_name]
        return kp, stats_df


def plotTrajectories(df, groupby="particle", t_st=0, t_end=1e5, doneTrajNoMarker=False,
                     startAtOrigin=False, dispLegend=False, useDrawnLines=False, drawnLines=[], hideLines=False,
                     axForVelocityPlot=[], **kwargs):
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

    if axForVelocityPlot == []:
        plotVelocity = False
    else:
        plotVelocity = True

    # if doneTrajNoMarker:
    df["t_end"] = df.groupby('particle')['frame'].transform(
        'last')  # mark end time for each particle trajectory

    df = df[(df.frame >= t_st) & (df.frame <= t_end)]  # filter by time

    if startAtOrigin:  # for each particle, subtract x[0] and y[0] so the traj starts at the origin
        df['x'] = df['x'] - df.groupby('particle')['x'].transform('first')
        df['y'] = df['y'] - df.groupby('particle')['y'].transform('first')

    particle_list = df[groupby].unique()
    gb = df.groupby(groupby)
    cmap_colors = plt.cm.get_cmap("tab10").colors
    N_colors = len(cmap_colors)

    if "state" in df.columns:
        multiColorLine = True
    else:
        multiColorLine = False

    if useDrawnLines:
        line_array = drawnLines[0]
        lc_array = drawnLines[1]
        scatter_array = drawnLines[2]
        vel_line_array = drawnLines[3]
        vel_lc_array = drawnLines[4]
        vel_scatter_array = drawnLines[5]

        if not hideLines:
            for l in line_array:
                if l is not None:
                    l.set_visible(True)
            for s in scatter_array:
                if s is not None:
                    s.set_visible(True)
            for c in lc_array:
                if c is not None:
                    c.set_visible(True)
            for l in vel_line_array:
                if l is not None:
                    l.set_visible(True)
            for s in vel_scatter_array:
                if s is not None:
                    s.set_visible(True)
            for c in vel_lc_array:
                if c is not None:
                    c.set_visible(True)
    else:
        line_array = np.empty(len(particle_list), dtype=object)
        scatter_array = np.empty(len(particle_list), dtype=object)
        lc_array = np.empty(len(particle_list), dtype=object)
        vel_line_array = np.empty(len(particle_list), dtype=object)
        vel_scatter_array = np.empty(len(particle_list), dtype=object)
        vel_lc_array = np.empty(len(particle_list), dtype=object)

    for i, particle_id in enumerate(particle_list):
        particle_df = gb.get_group(particle_id)
        x = particle_df["x"].values
        y = particle_df["y"].values

        if plotVelocity:
            t = particle_df["t"].values
            v = particle_df["dr2"].values / t
            # t = t[1:]
            # v = v[1:]

        if multiColorLine:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            if plotVelocity:
                vel_points = np.array([t, v]).T.reshape(-1, 1, 2)
                vel_segments = np.concatenate([vel_points[:-1], vel_points[1:]], axis=1)

            color1 = cmap_colors[i % N_colors]
            color2 = tuple(np.divide(color1, 2))  # make it darker
            cmap = ListedColormap([color1, color2])
            norm = BoundaryNorm([0, 0.6, 1], cmap.N)
            colorby = particle_df["state"].values

            if useDrawnLines:
                lc = lc_array[i]
                lc.set_segments(segments)
                lc.set_norm(norm)
                lc.set_array(colorby)
                if plotVelocity:
                    vel_lc = vel_lc_array[i]
                    vel_lc.set_segments(vel_segments)
                    vel_lc.set_norm(norm)
                    vel_lc.set_array(colorby)
            else:
                lc = LineCollection(segments, cmap=cmap, norm=norm, array=colorby)
                line = ax.add_collection(lc)
                line.set_label(groupby + ": " + str(particle_id))
                lc_array[i] = lc
                if plotVelocity:
                    vel_lc = LineCollection(vel_segments, cmap=cmap, norm=norm, array=colorby)
                    vel_line = axForVelocityPlot.add_collection(vel_lc)
                    vel_line.set_label(groupby + ": " + str(particle_id))
                    vel_lc_array[i] = vel_lc

            ax.autoscale()
            if useDrawnLines:
                scat = scatter_array[i]
                scat.set_offsets(np.vstack([x, y]).transpose())
                if plotVelocity:
                    vel_scat = vel_scatter_array[i]
                    vel_scat.set_offsets(np.vstack([t, v]).transpose())
            else:
                scat = ax.scatter(x, y, marker="x", c=np.floor(colorby), cmap=cmap)
                scatter_array[i] = scat
                if plotVelocity:
                    vel_scat = axForVelocityPlot.scatter(t, v, marker="o", c=np.floor(colorby), cmap=cmap)
                    vel_scatter_array[i] = vel_scat
            # Show scatter only if the following conditions aren't met
            if not ((particle_df["t_end"].head(1).values > t_end) or not doneTrajNoMarker):
                scat.set_visible(False)
                if plotVelocity:
                    vel_scat.set_visible(False)
        # not multicolor line
        else:
            if useDrawnLines:
                line = line_array[i]
                line.set_xdata(x)
                line.set_ydata(y)
                if plotVelocity:
                    vel_line = vel_line_array[i]
                    vel_line.set_xdata(t)
                    vel_line.set_ydata(v)
            else:
                vel_line = axForVelocityPlot.plot(t, v, label=(groupby + ": " + str(particle_id)), marker="o")
                vel_line_array[i] = vel_line[0]

    if dispLegend:
        leg = plt.legend(loc='upper left')
        # if plotVelocity:
        #     vel_leg = axForVelocityPlot.legend(loc='upper left')
        # ax.get_figure().canvas.draw()
    # ax.axis('square')
    # ax.set_aspect('equal', 'box')
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    if (doneTrajNoMarker) and (not multiColorLine) and (not hideLines):
        # get index of plots to remove markers from according to particles with trajs that dont last after t_end
        ind = np.where(np.isin(df.particle.unique(), df.particle[df.t_end <= t_end].unique()))[
            0]  # what plot indices to remove marker
        for i in ind:
            line_array[i].set_marker('None')
            if plotVelocity:
                vel_line_array[i].set_marker('None')

    # return all the produced graphic handles
    if hideLines:
        for l in line_array:
            if l is not None:
                l.set_visible(False)
        for s in scatter_array:
            if s is not None:
                s.set_visible(False)
        for c in lc_array:
            if c is not None:
                c.set_visible(False)
        for l in vel_line_array:
            if l is not None:
                l.set_visible(False)
        for s in vel_scatter_array:
            if s is not None:
                s.set_visible(False)
        for c in vel_lc_array:
            if c is not None:
                c.set_visible(False)

    return [line_array, lc_array, scatter_array, vel_line_array, vel_lc_array, vel_scatter_array]


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
    ap.add_argument("-o", "--output", required=False, default=(output_name + '.mp4'),
                    help="output video file")

    import sys
    sys.argv = ['']
    del sys

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
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Be sure to use lower case

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


def animate_Trajectories(df, ax=[], t_end_array=np.arange(0, 201), startAtOrigin=True, preStr="traj",
                         video_name="traj_video", gif_name="traj_gif", showFigure=False, saveFiles=True,
                         dispLegend=False, doneTrajNoMarker=True, fps=0., axForVelocityPlot=[]):
    files_path = IMG4VID_DIR + "\\"
    if ax == []:
        fig, ax = plt.subplots(figsize=(6, 6))
    max_frame = t_end_array[-1]
    # first plot the lines, then we only modify them for better runtime
    drawnLines = plotTrajectories(df, t_end=max_frame, doneTrajNoMarker=doneTrajNoMarker,
                                  startAtOrigin=startAtOrigin,
                                  dispLegend=False, ax=ax, useDrawnLines=False, drawnLines=[],
                                  hideLines=True, axForVelocityPlot=axForVelocityPlot)
    if showFigure:
        fig.canvas.draw()
        plt.show(block=False)
    filenames = []
    for i, t_end in enumerate(t_end_array):

        notebookAnimateTraj(df, ax, t_end, max_frame, startAtOrigin=startAtOrigin, doneTrajNoMarker=doneTrajNoMarker,
                            dispLegend=dispLegend, useDrawnLines=True, drawnLines=drawnLines, fps=fps,
                            axForVelocityPlot=axForVelocityPlot)
        if showFigure:
            fig.canvas.draw()
            fig.canvas.flush_events()
        # create file name and append it to a list
        filename = files_path + f'{preStr}{i:04d}.png'
        filenames.append(filename)
        # save frame
        if saveFiles:
            plt.savefig(filename, format="png", dpi=400)

    if saveFiles:
        createVideo(path=files_path, fps=20.0, output_name=OUTPUTS_DIR + "\\" + video_name, image_type="png")
        createGif(path=files_path, buffer=5, output_name=OUTPUTS_DIR + "\\" + gif_name, image_type="png")
        for f in os.listdir(files_path):
            os.remove(files_path + f)


def animate_G_dx_dt(df, d_array=np.arange(1, 101), group_by="experiment", x="xy", kind="kde", video_name="myvideo",
                    gif_name="mygif", isdfPadded=False,
                    var_name="myvar", preStr="", compare_conv=False, xlim=20., ylim=[1e-6, 1.], semilogscale=True,
                    saveFiles=True, showFigures=False, **kwargs):
    # Create gif and mp4 and variance plot of G(delta x/y, delta t)
    # Inputs:  1. dataframe
    #          2. d_array - delta t array in frame units
    #          3. x = "x" or "y"
    #          4. mode = "kde" or "hist"
    #          5. compare_conv = also plot convolution of the first distribution with itself
    # Outputs: 1. mp4 file with G(dx,dt) animation
    #          2. gif file with G(dx,dt) animation
    #          3. png file variance and counts vs dt (var is solid line, counts dashed)

    # obtain x,y for convolutions
    if compare_conv:
        groups = df["experiment"].unique().tolist()
        xdata = []
        ydata_init = []
        ydata_fourier_init = []
        ydata_fourier = []
        ydata = []
        # colors = sns.color_palette("hls", len(df.groupby(group_by)))
        colors = sns.color_palette("tab10", len(df.groupby(group_by)))
        for i, g in enumerate(groups):
            kp_stam = plot_G_dx_dt(df.groupby(group_by).get_group(g), dt=int(d_array[0]), direction=x,
                                   group_by=group_by, kind="kde", gridsize=10 ** 4, isdfPadded=isdfPadded,
                                   filter_diff=0.0, return_stats=False, semilogscale=semilogscale, **kwargs)  # no ax

            lines = plt.gca().get_lines()
            L = lines[0]
            xdata.append(L.get_xdata())
            ydata_init.append(L.get_ydata() / np.trapz(L.get_ydata(), L.get_xdata()))  # normalize
            colors.append(L.get_color())

        # now add padding
        for i, X in enumerate(xdata):
            xrange = max(X) - min(X)
            dx = xrange / len(X)
            center = X[ydata_init[i].argmin()]
            XX = np.arange(-xlim * 10 + center, xlim * 10 + center + 1, step=dx)  # symmetric array
            YY = 0 * XX
            ind_start = np.argmin(np.abs(XX - X[0]))
            for k, Xk in enumerate(X):
                YY[ind_start + k] = ydata_init[i][k]
            xdata[i] = XX
            ydata_init[i] = YY
            ydata_fourier_init.append(np.fft.fft(YY))
        plt.close('all')

        def gauss(xi, a, x0, sigma):
            return a * np.exp(-(xi - x0) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    # now we really plot
    fig = plt.figure()
    ax = plt.gca()

    if showFigures:
        fig.canvas.draw()
        plt.show(block=False)
    filenames = []
    files_path = IMG4VID_DIR + "\\"
    for i, d in enumerate(d_array):
        plt.cla()
        if kind == "kde":
            ax.set_ylim(ylim)
        kp, cur_stats_df = plot_G_dx_dt(df, dt=d, ax=ax, direction=x, group_by=group_by, kind=kind,
                                        isdfPadded=isdfPadded,
                                        filter_diff=0.0, return_stats=True, semilogscale=semilogscale, **kwargs)
        ax.set_title(fr'Displacement distribution $G(\Delta {x},\Delta t= {d}$' + " frames)")
        ax.set_xlim([-xlim, xlim])
        ax.set_ylim(ylim)
        if compare_conv:
            for j, group in enumerate(groups):
                if i == 0:
                    ydata.append(ydata_init[j])
                    ydata_fourier.append(ydata_fourier_init[j])
                    center_gaussian = 0
                else:
                    ydata_fourier[j] = ydata_fourier[j] * ydata_fourier_init[j]
                    ydata[j] = np.real(np.fft.ifft(ydata_fourier[j]))
                    ydata[j] = np.maximum(ydata[j], 0)  # its ok to discard imaginary and take positive
                    if i % 2 == 0:
                        ydata[j] = np.fft.fftshift(ydata[j])
                    ydata[j] = ydata[j] / np.trapz(ydata[j], xdata[j])  # normalize
                    # now center around 0
                    popt, pcov = scipy.optimize.curve_fit(gauss, xdata[j], ydata[j],
                                                          p0=[1, xdata[j][np.argmax(ydata[j])], np.sqrt(1 + i)])
                    center_gaussian = popt[1]
                plt.plot(xdata[j] - center_gaussian, ydata[j], "--", color=colors[j])

        if showFigures:
            fig.canvas.draw()
            fig.canvas.flush_events()
        # create file name and append it to a list
        filename = files_path + f'{i:04d}.png'
        filenames.append(filename)
        if saveFiles:
            # save frame
            plt.savefig(filename, format="png", dpi=400)
            # plt.savefig(".\\for_latex\\" + f'{preStr}{i}.png', format="png", dpi=400)
            # plt.savefig("C:\\Users\\Amit\\OneDrive - mail.tau.ac.il\\for_latex\\" + f'{preStr}{i}.png', format="png", dpi=400)
            # handle the stats df
            if i == 20:
                stats_df = cur_stats_df
            elif i > 20:
                stats_df = stats_df.append(cur_stats_df)

    if saveFiles:
        createVideo(path=files_path, fps=6.0, output_name=OUTPUTS_DIR + "\\" + video_name, image_type="png")
        # createGif(path=files_path, buffer=5, output_name="./files/" + gif_name, image_type="png")
        for f in os.listdir(files_path):
            os.remove(files_path + f)

        if len(stats_df) > 0:
            backupBeforeSave(OUTPUTS_DIR + "\\stats_df.fea")
            stats_df.reset_index().to_feather(OUTPUTS_DIR + "\\stats_df.fea")
            stats_df = stats_df.reset_index()
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
            var_name_full = OUTPUTS_DIR + "\\" + var_name + ".png"
            backupBeforeSave(var_name_full)
            plt.savefig(var_name_full, format="png", dpi=400)
    return


def showTrajLengthStats(df):
    for i, exp in enumerate(df.experiment.unique()):
        d = df.groupby("experiment").get_group(exp).groupby("particle").apply(len).to_frame()
        d = d.rename(columns={0: "len"})
        d.loc[:, "experiment"] = exp
        if i == 0:
            D = d
        else:
            D = D.append(d)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xscale('log')
    # ax.set_yscale('log')
    sns.histplot(data=D, x="len", hue="experiment", ax=ax)
    plt.show()


def trajAxesLimits(df, startAtOrigin):
    if startAtOrigin:
        lim = 1.05 * max(
            (df['x'] - df.groupby('particle')['x'].transform('first')).abs().max(),
            (df['y'] - df.groupby('particle')['y'].transform('first')).abs().max())
        xlim = [-lim, lim]
        ylim = [-lim, lim]
    else:
        xlim = [df['x'].min(), df['x'].max()]
        ylim = [df['y'].min(), df['y'].max()]
        # to make axes square
        if (xlim[1] - xlim[0]) > (ylim[1] - ylim[0]):
            delta = (xlim[1] - xlim[0]) / 2
            center = (ylim[0] + ylim[1]) / 2
            ylim[0] = center - delta
            ylim[1] = center + delta
        else:
            delta = (ylim[1] - ylim[0]) / 2
            center = (xlim[0] + xlim[1]) / 2
            xlim[0] = center - delta
            xlim[1] = center + delta

    return xlim, ylim


def stateTimesHistogram(df=pd.DataFrame([]), times_df=pd.DataFrame([]), kind="kde", eqWeightParticles=True,
                        log_scale=True, ax=[], saveFiles=False):
    if times_df.empty:  # create times_df from the normal input df
        experiments = df.experiment.unique()
        gb_exp = df.groupby("experiment")
        experiment_arr = np.asarray([])
        particle_arr = np.asarray([])
        t = np.asarray([])
        weight = np.asarray([])
        state = np.asarray([])

        for i, experiment in enumerate(experiments):
            free_time = np.asarray([])
            free_particle = np.asarray([])
            free_weight = np.asarray([])
            free_experiment = np.asarray([])
            stuck_time = np.asarray([])
            stuck_particle = np.asarray([])
            stuck_weight = np.asarray([])
            stuck_experiment = np.asarray([])
            free_state = np.asarray([])
            stuck_state = np.asarray([])
            df_exp = gb_exp.get_group(experiment)
            particles = df_exp.particle.unique()
            gb_particle = df_exp.groupby("particle")
            for j, particle in enumerate(particles):
                df_particle = gb_particle.get_group(particle)
                switching_times = df_particle[df_particle["state"] == 0.5]
                time_spent = switching_times.t.diff()[1:].values
                init_state = df_particle.state[switching_times.index[0] + 1]
                even = time_spent[::2]
                odd = time_spent[1::2]
                if init_state == 0:  # start free
                    time_spent_free = even
                    time_spent_stuck = odd
                else:
                    time_spent_free = odd
                    time_spent_stuck = even

                len_free = len(time_spent_free)
                len_stuck = len(time_spent_stuck)
                free_time = np.append(free_time, time_spent_free)
                stuck_time = np.append(stuck_time, time_spent_stuck)
                free_particle = np.append(free_particle, particle * np.ones(len_free + 1))
                stuck_particle = np.append(stuck_particle, particle * np.ones(len_stuck))
                free_weight = np.append(free_weight, np.ones(len_free) / len_free)
                stuck_weight = np.append(stuck_weight, np.ones(len_stuck) / len_stuck)
                free_experiment = np.append(free_experiment, np.repeat(experiment, len_free + 1))
                stuck_experiment = np.append(stuck_experiment, np.repeat(experiment, len_stuck))
                free_state = np.append(free_state, np.repeat("free", len_free))
                stuck_state = np.append(stuck_state, np.repeat("stuck", len_stuck))
                # add the total traj duration
                free_time = np.append(free_time, df_particle.trajDurationSec.values[0])
                free_weight = np.append(free_weight, 0.)
                free_state = np.append(free_state, "trajLength")

            free_weight = free_weight / sum(free_weight)
            stuck_weight = stuck_weight / sum(stuck_weight)
            experiment_arr = np.append(experiment_arr, np.append(free_experiment, stuck_experiment))
            particle_arr = np.append(particle_arr, np.append(free_particle, stuck_particle))
            t = np.append(t, np.append(free_time, stuck_time))
            weight = np.append(weight, np.append(free_weight, stuck_weight))
            state = np.append(state, np.append(free_state, stuck_state))

        weight[weight == 0.] = 1.  # for the trajLen
        d = {"experiment": experiment_arr, "particle": particle_arr, "t": t, "weights": weight, "state": state}
        times_df = pd.DataFrame(data=d).sort_values(["experiment", "particle", "state"])
        times_df.index = np.arange(len(times_df))
        times_df["experiment+state"] = times_df["experiment"] + " ; " + times_df["state"]

    if len(ax) == 0:
        ax = plt.gca()
    if kind == "kde":
        if log_scale:
            clip = (-10., 1000.)
        else:
            clip = (0., 1000.)
        if eqWeightParticles:
            kp = sns.kdeplot(data=times_df, x="t", hue=("experiment+state"), weights="weights",
                             hue_order=["state", "experiment"],
                             common_norm=False, clip=clip, log_scale=[log_scale, log_scale], ax=ax)
        else:
            kp = sns.kdeplot(data=times_df, x="t", hue=("experiment+state"),
                             hue_order=["state", "experiment"],
                             common_norm=False, clip=clip, log_scale=[log_scale, log_scale], ax=ax)
    elif kind == "hist":
        if eqWeightParticles:
            kp = sns.histplot(data=times_df, x="t", hue="experiment+state", weights="weights", bins=100,
                              common_norm=False, stat="density", log_scale=[log_scale, log_scale], ax=ax)
        else:
            kp = sns.histplot(data=times_df, x="t", hue="experiment+state",
                              common_norm=False, stat="density", log_scale=[log_scale, log_scale], ax=ax)

    if saveFiles:
        filepath = OUTPUTS_DIR + "\\times_df.fea"
        extras.backupBeforeSave(filepath)
        times_df.to_feather(filepath)

    return times_df


def quickTraj(df, N_particles=1, t_end_array=np.arange(0, 101), ):
    random_particle_id = np.random.choice(df.particle.unique(), size=N_particles, replace=False)
    small_df = df[df["particle"].isin(random_particle_id)]
    ax = plt.gca()
    print("Mean dr^2: " + str(small_df.dr2.dropna().mean()))
    plot_MSD(small_df, lagtime=np.asarray([1., 2., 5., 10., 50.]), group_by="particle", ax=ax)
    animate_G_dx_dt(small_df, d_array=t_end_array, showFigures=False, saveFiles=False, group_by="particle")
    ax = plt.gca()
    ax.set_xlim([-15, 15])
    # ax.set_yscale("log")
    # ax.set_ylim([0.0001, 1])
    ax.set_yscale("linear")
    ax.set_ylim([0, 1])
    out = getDiffs(small_df, 1)
    dx = out[0]
    dy = out[1]
    print("Mean dx,dy,dr2 correct: " + str(np.mean(np.abs(dx))) + ", " + str(np.mean(np.abs(dy))) + ", " + str(
        np.mean(dx ** 2 + dy ** 2)))
    animate_Trajectories(small_df, showFigure=True, t_end_array=np.arange(0, 101),
                         saveFiles=False, dispLegend=True)
    plt.show()


def notebookAnimateTraj(df, ax, t_end, max_frame, startAtOrigin=True, doneTrajNoMarker=True, dispLegend=False,
                        useDrawnLines=False, drawnLines=[], fps=0., axForVelocityPlot=[]):
    # ax.cla()
    out_graphic_handles = plotTrajectories(df, t_end=t_end, doneTrajNoMarker=doneTrajNoMarker,
                                           startAtOrigin=startAtOrigin,
                                           dispLegend=False, ax=ax, useDrawnLines=useDrawnLines,
                                           drawnLines=drawnLines, axForVelocityPlot=axForVelocityPlot)
    if dispLegend:
        try:
            ax.get_figure().canvas.draw()
        except:
            pass
        leg = plt.legend(loc='upper left')
        plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text

    xlim, ylim = trajAxesLimits(df[df["frame"] <= max_frame], startAtOrigin)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if fps == 0:
        ax.set_title(f'$t = {t_end}$ frames')
    else:
        ax.set_title(f'$t$ = {int(t_end * 1000 / fps)} msec')
    if startAtOrigin:
        ax.set_xlabel(r'$x-x_{0} [\mu {\rm m}]$')
        ax.set_ylabel(r'$y-y_{0} [\mu {\rm m}]$')
    else:
        ax.set_xlabel(r'$x [\mu {\rm m}]$')
        ax.set_ylabel(r'$y [\mu {\rm m}]$')

    if not axForVelocityPlot == []:
        axForVelocityPlot.set_ylabel(r'$\Delta r^{2} / \Delta t \,[\mu {\rm m}^{2} / sec]$')
        axForVelocityPlot.set_xlabel(r'$t [sec]$')

    return out_graphic_handles
