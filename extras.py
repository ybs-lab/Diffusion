import cProfile
import pstats
import io
import os
import numpy as np
import pandas as pd
from datetime import datetime
import ctypes


def profile(mode, profiler=[], filename='profile_results'):
    if mode.lower() == "on":
        pr = cProfile.Profile()
        pr.enable()
        return pr
    elif mode.lower() == "off" or mode.lower() == "viewer":
        pr = profiler
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        with open((filename + '.txt'), 'w+') as f:
            f.write(s.getvalue())
        pr.dump_stats((filename + '.prof'))
        print("For a good ui view, execute in terminal: snakeviz .\\" + filename + ".prof")


def nowString():
    return datetime.now().strftime('%y_%m_%d__%H_%M_%S')


def backupBeforeSave(filepath):
    # If a file exists, back it up by renaming it with the current date and time
    # Input: 1. filepath = full path to file, including suffix of file
    # Output: Renames the file if it exists (else does nothing)
    if os.path.isfile(filepath):  # if a file exists with the same name, rename existing file
        address_filename = os.path.split(filepath)
        os.rename(filepath,
                  address_filename[0] + "/" + nowString() + address_filename[1])


def notebookSetupWidgets(widgets):
    # I created this ugly function so the notebook will look less clustered. Input is the imported widgets class.
    expSelectWidget = widgets.SelectMultiple(options=[], layout={'width': 'max-content'},
                                             description='Select Experiment(s)',
                                             style={'description_width': 'initial'}, disabled=False)
    rdmEnableWidget = widgets.Checkbox(value=True, description='Select random particles', disabled=False,
                                       indent=False)
    rdmSelectionWidget = widgets.BoundedIntText(value=5, min=1, max=100, step=1,
                                                description='Number of random particles to select:',
                                                style={'description_width': 'initial'}, disabled=False)
    specificEnableWidget = widgets.Checkbox(value=False, description='Select a specific particle', disabled=True,
                                            indent=False)
    specificParticleSelectionWidget = widgets.BoundedIntText(min=0, max=1, step=1, description='Particle ID:',
                                                             style={'description_width': 'initial'}, disabled=True)

    def rdmEnableObserve(
            change):
        isChecked = rdmEnableWidget.value;
        rdmSelectionWidget.disabled = not isChecked;
        specificParticleSelectionWidget.disabled = isChecked;
        specificEnableWidget.disabled = isChecked

    def rdmSpecificObserve(
            change):
        isChecked = specificEnableWidget.value;
        expSelectWidget.disabled = isChecked;
        rdmEnableWidget.disabled = isChecked;
        rdmSelectionWidget.disabled = (
                isChecked or not rdmEnableWidget.value);
        specificParticleSelectionWidget.disabled = not isChecked

    rdmEnableWidget.observe(rdmEnableObserve, names='value');
    specificEnableWidget.observe(rdmSpecificObserve, names='value')


    #handle button interactions
    buttonFilterWidget = widgets.Button(description="Filter!")

    def filterDataframesFun(full_df,full_padded_df,df_to_change, padded_df_to_change):
        if specificEnableWidget.value:
            df = full_df[full_df["particle"] == specificParticleSelectionWidget.value]
            padded_df = full_padded_df[full_padded_df["particle"] == specificParticleSelectionWidget.value]
        else:
            df = full_df[np.isin(full_df["experiment"], expSelectWidget.value)]
            padded_df = full_padded_df[np.isin(full_padded_df["experiment"], expSelectWidget.value)]
            if rdmEnableWidget.value:
                all_particles = df.particle.unique()
                random_particle_id = np.random.choice(all_particles,
                                                      size=min(rdmSelectionWidget.value, len(all_particles)),
                                                      replace=False)
                df = full_df[full_df["particle"].isin(random_particle_id)]
                padded_df = full_padded_df[full_padded_df["particle"].isin(random_particle_id)]
        # Now convert the df_to_change, padded_df_to_change:
        # First drop all the rows
        df_to_change.drop(df_to_change.index, inplace=True)
        padded_df_to_change.drop(padded_df_to_change.index, inplace=True)
        # Now fill the columns:
        allCols_df = np.asarray(df.columns)
        for col in allCols_df:
            df_to_change[col] = df[col]
        allCols_padded_df = np.asarray(padded_df.columns)
        for col in allCols_padded_df:
            padded_df_to_change[col] = padded_df[col]

    return expSelectWidget, rdmEnableWidget, rdmSelectionWidget, specificEnableWidget, specificParticleSelectionWidget,buttonFilterWidget,filterDataframesFun


def separateStuckFree(df):
    df_free = df.copy()
    df_free["experiment"] = df_free["experiment"] + "_free"
    df_free["particle"] = df_free["particle"] + 1000000
    df_free = df_free[df_free["state"] != 1]
    df_stuck = df.copy()
    df_stuck["experiment"] = df_stuck["experiment"] + "_stuck"
    df_stuck["particle"] = df_stuck["particle"] + 2000000
    df_stuck = df_stuck[df_stuck["state"] == 1]
    df = df.append(df_free.append(df_stuck))
    return df
