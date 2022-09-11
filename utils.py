import cProfile
import pstats
import io
import os
import numpy as np
from itertools import groupby
from datetime import datetime
import scipy.ndimage.measurements as mnts


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
        print(
            "For a good ui view, execute in terminal: snakeviz ./" + filename + ".prof" + "\nOr go to https://nejc.saje.info/pstats-viewer.html")


def now_string():
    return datetime.now().strftime('%y_%m_%d__%H_%M_%S')


def backup_before_save(filepath):
    # If a file exists, back it up by renaming it with the current date and time
    # Input: 1. filepath = full path to file, including suffix of file
    # Output: Renames the file if it exists (else does nothing)
    if os.path.isfile(filepath):  # if a file exists with the same name, rename existing file
        address_filename = os.path.split(filepath)
        os.rename(filepath,
                  os.path.join(address_filename[0], now_string() + "_" + address_filename[1]))


def notebook_setup_widgets(widgets):
    # I created this ugly function so the notebook will look less clustered. Input is the imported widgets class.
    expSelectWidget = widgets.SelectMultiple(options=[], layout={'width': 'max-content'},
                                             description='Select Experiment(s)',
                                             style={'description_width': 'initial'}, disabled=False)
    rdmEnableWidget = widgets.Checkbox(value=True, description='Select random particles', disabled=False,
                                       indent=False)
    rdmSameVideoWidget = widgets.Checkbox(value=True, description='From the same video?', disabled=False,
                                          indent=False)
    keepStuckWidget = widgets.Checkbox(value=True, description='Keep stuck particles?', disabled=False,
                                       indent=False)
    rdmSelectionWidget = widgets.BoundedIntText(value=5, min=1, max=500, step=1,
                                                description='Number of random particles to select:',
                                                style={'description_width': 'initial'}, disabled=False)
    specificEnableWidget = widgets.Checkbox(value=False, description='Select a specific particle', disabled=True,
                                            indent=False)
    specificParticleSelectionWidget = widgets.BoundedIntText(min=0, max=1, step=1, description='Particle ID:',
                                                             style={'description_width': 'initial'}, disabled=True)

    def rdmEnableObserve(change):
        isChecked = rdmEnableWidget.value
        rdmSelectionWidget.disabled = not isChecked
        rdmSameVideoWidget.disabled = not isChecked
        specificParticleSelectionWidget.disabled = isChecked
        specificEnableWidget.disabled = isChecked

    def rdmSpecificObserve(change):
        isChecked = specificEnableWidget.value
        expSelectWidget.disabled = isChecked
        rdmEnableWidget.disabled = isChecked
        keepStuckWidget.disabled = isChecked
        rdmSelectionWidget.disabled = (
                isChecked or not rdmEnableWidget.value)
        specificParticleSelectionWidget.disabled = not isChecked

    rdmEnableWidget.observe(rdmEnableObserve, names='value')
    specificEnableWidget.observe(rdmSpecificObserve, names='value')

    # handle button interactions
    buttonFilterWidget = widgets.Button(description="Filter!")

    def filterDataframesFun(full_df, df_to_change):
        if specificEnableWidget.value:
            df = full_df[full_df["particle"] == specificParticleSelectionWidget.value]
        else:
            if not keepStuckWidget.value:
                full_df = full_df[full_df.isMobile]
            selectedExperiments = expSelectWidget.value
            if len(selectedExperiments) == 0:  # nothing selected = all selected
                selectedExperiments = full_df.experiment.unique()

            df = full_df[np.isin(full_df["experiment"], selectedExperiments)]
            if rdmEnableWidget.value:
                if rdmSameVideoWidget.value:
                    all_experiments = df.experiment.unique()
                    df = df[df.experiment == np.random.choice(all_experiments)]
                    all_filenames = df.filename.unique()
                    df = df[df.filename == np.random.choice(all_filenames)]
                # possibly restricted particles to be in the same experiments
                all_particles = df.particle.unique()
                random_particle_id = np.random.choice(all_particles,
                                                      size=min(rdmSelectionWidget.value, len(all_particles)),
                                                      replace=False)
                df = full_df[full_df["particle"].isin(random_particle_id)]
        # Now convert the df_to_change, padded_df_to_change:
        # First drop all the rows
        df_to_change.drop(df_to_change.index, inplace=True)
        # Now fill the columns:
        allCols_df = np.asarray(df.columns)
        for col in allCols_df:
            df_to_change[col] = df[col]

    return expSelectWidget, rdmEnableWidget, rdmSameVideoWidget, keepStuckWidget, rdmSelectionWidget, specificEnableWidget, specificParticleSelectionWidget, buttonFilterWidget, filterDataframesFun


def arr2string(int_array, delimiter='_'):
    return delimiter.join(str(e) for e in int_array)


def string2arr(string, delimiter='_'):
    # return [int(e) for e in string.split(delimiter)]
    return [int(e) if e.isdigit() else e for e in string.split(delimiter)]


def add_leading_zeros(path, n_digits=6):
    for f in os.listdir(path):
        filename, extension = os.path.splitext(f)
        sum = 0
        n_last_digit = 0
        for n, char in enumerate(filename[::-1]):
            if char.isdigit():
                sum += int(char) * 10 ** n
                n_last_digit += 1
            else:
                break
        if n_last_digit <= n_digits:
            new_f = filename[0:(len(filename) - n_last_digit)] + str(sum).zfill(
                n_digits) + extension  # not -1 because it won't work for n_last_digit=0
            os.rename(os.path.join(path, f), os.path.join(path, new_f))


def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    maxVal = x.max()
    exp_arr = np.zeros(x.shape, dtype=object)
    for i, xval in enumerate(x):
        exp_arr[i] = np.exp(xval - maxVal)
    LSE = maxVal + np.log(exp_arr.sum())
    return LSE


def vec_translate(a, my_dict):
    # From https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    return np.vectorize(my_dict.__getitem__)(a)


def isempty(x):
    if np.asarray(x).size == 0:
        return True
    else:
        return False


def index_intervals_extract(iterable):
    # From https://www.geeksforgeeks.org/python-make-a-list-of-intervals-with-sequential-numbers/
    iterable = sorted(set(iterable))
    for key, group in groupby(enumerate(iterable),
                              lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def isnotebook():
    # from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def arr_of_length_of_true_segments(arr):
    # from https://stackoverflow.com/questions/49776310/numpy-2d-boolean-array-count-consecutive-true-sizes
    labeled, clusters = mnts.label(arr)
    sizes = mnts.sum(arr, labeled, index=range(clusters + 1))
    return sizes.astype(int)[1:]


def accuracy_of_hidden_paths(S, S_true, XT, XT_true):
    N = len(S_true)
    XT = np.reshape(XT, [N, 2])
    XT_true = np.reshape(XT_true, [N, 2])
    accuracy = np.sum((S_true == S) * ((S == 0) + (S == 1) * np.prod(np.isclose(XT, XT_true, rtol=1e-05, atol=1e-08, equal_nan=True),
                                                   axis=1))) / float(N)
