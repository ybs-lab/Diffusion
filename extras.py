import cProfile
import pstats
import io
import os
from datetime import datetime


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


def backupBeforeSave(filepath):
    # If a file exists, back it up by renaming it with the current date and time
    # Input: 1. filepath = full path to file, including suffix of file
    # Output: Renames the file if it exists (else does nothing)
    if os.path.isfile(filepath):  # if a file exists with the same name, rename existing file
        address_filename = os.path.split(filepath)
        os.rename(filepath,
                  address_filename[0] + "/" + datetime.now().strftime('%d_%m__%H_%M_%S_') + address_filename[1])
