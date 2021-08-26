import cProfile
import pstats
import io


def profile(mode, profiler, filename='test.txt'):
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
        with open(filename, 'w+') as f:
            f.write(s.getvalue())
