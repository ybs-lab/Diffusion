import numpy as np
# import autograd.numpy as np
# from autograd import grad, jacobian, hessian
# from autograd.extend import primitive, defvjp
import itertools
from multiprocessing import Pool
import scipy.optimize
import time
from extras import backupBeforeSave
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.special import factorial, comb


def estSpatialProb(dt, dr2, x, y, regime, D, A):
    if regime == 0:  # 0=free 1=stuck
        log_p = estFreeProb(dt, dr2, D)
        log_p = log_p
    else:
        log_p = estStuckProb(x, y, A)
        log_p = log_p

    return log_p


def estFreeProb(dt, dr2, D):
    # Receive sorted (dx^2+dy^2) and dt for a single particle and estimate the probably density for the free regime
    # One parameter D is diffusion coefficient
    # dx = df.dx.values
    # dy = df.dy.values
    # dt = df.dt.values

    # dt = np.diff(df.t.values)
    # dx = np.diff(df.x.values)
    # dy = np.diff(df.y.values)

    log_p = -np.sum(dr2 / (4 * D * dt) + np.log((np.pi * 4 * D * dt)))

    return log_p


def estStuckProb(x, y, A):
    # Receive sorted x,y,t for a single particle and estimate the probably density for the stuck regime
    # One parameter A = mw^2 = 2kbT / k = variance of Gaussian of tethering (area units)

    # think about this more if this makes sense
    [x_center, y_center] = np.mean([x, y], axis=1)

    log_p = -np.sum(((x - x_center) ** 2 + (y - y_center) ** 2)) / (A) - len(x) * np.log(np.pi * A)
    return log_p


def estStickUnstickProb(init_state, t_stick_unstick_with_boundary, T_stick, T_unstick):
    # Receive time of stick and unstick events as well as initial time and end time of trajectory.
    # Two Parameters T_stick and T_unstick are the characteristic stick/unstick times
    dt = np.diff(t_stick_unstick_with_boundary)
    len_even = len(dt[::2])
    len_odd = len(dt[1::2])
    if init_state == 0:  # 0=free 1=stuck
        # even dt are stick events, odd are unstick
        p = np.exp(-sum(dt[::2]) / T_stick) / (T_stick ** len_even) * np.exp(-sum(dt[1::2]) / T_unstick) / (
                T_unstick ** len_odd)
    else:
        # even dt are unstick events, odd are stick
        p = np.exp(-sum(dt[::2]) / T_unstick) / (T_unstick ** len_even) * np.exp(-sum(dt[1::2]) / T_stick) / (
                T_stick ** len_odd)
    return p


def kEventsProb(K, init_state, T, T_stick, T_unstick):
    # K - number of events of stick/unstick when beginning at init_state, total time interval of T
    # don't confuse with k index of summation

    if np.abs(T_stick - T_unstick) < 0.001:  # simply Poisson
        P = (T / T_stick) ** (K) * np.exp(-T / T_stick) / factorial(K)

    else:
        if init_state == 0:  # 0=free 1=stuck
            t1 = T_stick
            t2 = T_unstick
        else:
            t1 = T_unstick
            t2 = T_stick
        r1 = 1 / t1
        r2 = 1 / t2
        rate = r1 - r2
        X = -rate * T

        if K == 0:
            P = np.exp(-T / t1)
        else:
            if K % 2 == 0:
                m = int(K / 2)
                P = 0
                for k in range(m):
                    p = 0
                    for n in range(m + k + 1):
                        p = p + (-X) ** n / factorial(n)
                    p = p * np.exp(-T / t1) - 1 * np.exp(-T / t2)
                    P = P + p * comb(m + k, k) * X ** (m - 1 - k) / factorial(m - 1 - k)
                P = (-1) ** m * P * (t1 * t2 / (t1 - t2) ** 2) ** m
            else:
                m = int((K + 1) / 2)
                P = 0
                for k in range(m):
                    p = 0
                    for n in range(m + k):
                        p = p + (-X) ** n / factorial(n)
                    p = p * np.exp(-T / t1) - 1 * np.exp(-T / t2)
                    P = P + p * comb(m - 1 + k, k) * X ** (m - 1 - k) / factorial(m - 1 - k)
                P = (-1) ** m * P * r1 * (r1 * r2) ** (m - 1) / (r1 - r2) ** (2 * m - 1)

    # TEMPORARY FIX TO UNDERFLOW PROBLEMS:
    if P < 0 and K > 10:
        P = kEventsProb(K - 2, init_state, T, T_stick, T_unstick)

    return P


def specificTrajectoryLikelihood(trajData, init_state, t_stick_unstick, dt_arr_jac, T_stick, T_unstick,
                                 D,
                                 A):
    t = trajData[0]
    dt = trajData[1]
    dr2 = trajData[2]
    x = trajData[3]
    y = trajData[4]
    t_stick_unstick_with_boundary = np.zeros(len(t_stick_unstick) + 2)
    t_stick_unstick_with_boundary[1:-1] = t_stick_unstick
    t_stick_unstick_with_boundary[-1] = t[-1]
    t_stick_unstick_with_boundary
    t_stick_unstick_with_t_f = t_stick_unstick_with_boundary[1:]

    # think about this more
    if len(t_stick_unstick_with_boundary) > 2:
        logp_temporal = np.log(
            estStickUnstickProb(init_state, t_stick_unstick_with_boundary, T_stick=T_stick,
                                T_unstick=T_unstick))
    else:
        logp_temporal = 0

    regime = init_state
    prev_t = t_stick_unstick_with_boundary[0]
    logp_spatial = 0
    for t_switch in t_stick_unstick_with_t_f:
        ind = (t >= prev_t) & (t <= t_switch)  # think about <= or <... closed or semiclosed intervals???
        if len(dt) < len(t):
            ind_no_st = ind[1:]
        else:
            ind_no_st = ind
        logp_spatial = logp_spatial + estSpatialProb(dt[ind_no_st], dr2[ind_no_st], x[ind], y[ind],
                                                     regime=regime, D=D, A=A)
        prev_t = t_switch
        regime = 1 - regime

    log_likelihood = (logp_spatial + logp_temporal + np.sum(
        np.log(dt_arr_jac))).astype(float)
    return log_likelihood


def particleLikelihood(df_in, T_stick, T_unstick, D, A, mode):
    # mode = 0 - returns just particle likelihood
    # mode = 1 - returns modified df with max-likelihood stick and unstick times
    K_event_range = range(5)
    init_states = [0, 1]  # [0, 1]
    p_initstate_free = 0.5  # prior for init state
    stepSize = int(2)  # can't be less than 2!
    minStepsForMode = int(4)  # can't be less than 1!

    # for convenience,make sure even number of samples
    df = df_in.copy()
    if len(df) % 2:
        df = df[:-1]
        trimmedLastTime = True
    else:
        trimmedLastTime = False

    t = df.t.values  # array of times
    len_t = len(t)
    trajData = extract_trajData_from_df(df)  # this eliminates the need for df (faster calculations)

    t_i = t[0]
    t_f = t[-1]
    T = df.dt.sum()  # not t_f-t_i because we sometimes include t_i for split dataframes TBD
    intermediate_times = t[
        np.arange(stepSize, len_t - stepSize, step=stepSize, dtype=int)]  # this is not affected by undersampling
    len_intermediate_times = len(intermediate_times)
    times_without_last = t[np.arange(0, len_t - stepSize, step=stepSize, dtype=int)]
    dt_intermediate_times = np.diff(times_without_last)  # for jacobian after

    logP_k_mat = np.zeros([2, len(K_event_range)], dtype=object)
    log_integral_mat = np.zeros([2, len(K_event_range)], dtype=object)
    max_likelihood_config = np.empty([2, len(K_event_range)], dtype=object)

    for i, init_state in enumerate(init_states):  # 0=free 1=stuck
        for j, K in enumerate(K_event_range):
            logP_k_mat[i, j] = np.log(
                kEventsProb(K, init_state, T, T_stick=T_stick, T_unstick=T_unstick).astype(float)) \
                               * np.abs(init_state - p_initstate_free)  # involve init state prior??
            if K == 0:
                # no integral here
                logp_configuration = specificTrajectoryLikelihood(trajData, init_state,
                                                                  [],
                                                                  1.,  # no jacobian because of no integral
                                                                  T_stick, T_unstick, D, A)
            else:
                # notice jumps of step>=2 ! this is not affected by undersampling
                iters = itertools.combinations(
                    intermediate_times[np.arange(len(intermediate_times), dtype=int)], K)
                iter_list = list(iters)  # list of times! not indices
                N_iters = len(iter_list)
                iters = itertools.combinations(
                    np.arange(len(intermediate_times), dtype=int), K)  # iterable of indices! not time

                logp_iterations = np.zeros(N_iters, dtype=object)

                # integral
                for count, indices in enumerate(iters):
                    if ((np.diff(indices).min(initial=minStepsForMode) < minStepsForMode) or
                            (len_intermediate_times - np.min(indices) <= minStepsForMode)):
                        logp_iterations[count] = -np.inf
                    else:
                        # make this simpler later
                        indices = np.asarray(indices)  # .astype(int)
                        t_stick_unstick = intermediate_times[indices]
                        dt_for_jac = dt_intermediate_times[indices]

                        indices_with_boundary = np.zeros(len(indices) + 2)
                        indices_with_boundary[1:-1] = indices + 2
                        indices_with_boundary[-1] = (len_t - 1)
                        indices_with_boundary = np.asarray(indices_with_boundary).astype(int)
                        t_stick_unstick_with_boundary = t[indices_with_boundary]

                        logp_iterations[count] = specificTrajectoryLikelihood(trajData, init_state,
                                                                              t_stick_unstick,
                                                                              dt_for_jac,
                                                                              T_stick, T_unstick, D, A)
                        # done calculating specific iteration
                logp_configuration = logsumexp(logp_iterations)
                if K >= 1:
                    max_likelihood_config[i, j] = iter_list[logp_iterations.argmax()]
            log_integral_mat[i, j] = logp_configuration.astype(float)

    logLikelihoodMat = logP_k_mat + log_integral_mat

    if mode == 0:
        logp_particle = logsumexp((logLikelihoodMat).flatten())
        return logp_particle
    else:
        max_index = np.unravel_index(np.argmax(logLikelihoodMat), logLikelihoodMat.shape)
        max_likelihood = np.max(logLikelihoodMat)
        max_likelihood_configuration = max_likelihood_config[max_index]

        switch_times = max_likelihood_configuration
        initial_state = max_index[0]
        current_state = initial_state
        df["state"] = current_state  # placeholder, unless no switching
        prev_t = df["t"].min()
        if switch_times == None:
            switch_times = []
        switch_times_with_end = np.zeros(len(switch_times) + 1)
        switch_times_with_end[:-1] = switch_times
        switch_times_with_end[-1] = df["t"].max() + 1
        # print(switch_times)
        # print("Pk: " + str(logP_k_mat[max_index]))
        for k, t_switch in enumerate(switch_times_with_end):
            ind = (df["t"] >= prev_t) & (df["t"] < t_switch)
            df.loc[ind, "state"] = current_state
            df.loc[df.index[np.where(ind)[0].min()], "state"] = 0.5  # transitions
            prev_t = t_switch
            current_state = 1 - current_state

        if trimmedLastTime:
            lineToAdd = df_in.tail(1).copy()
            lineToAdd.loc[:, "state"] = df.tail(1).state.values
            df = df.append(lineToAdd)

        df.loc[df.index[-1], "state"] = 0.5  # last is 0.5

        return df, max_likelihood


def labeledParticleLikelihood(particle_df, T_stick, T_unstick, D, A):
    trajData = extract_trajData_from_df(particle_df)  # this eliminates the need for df (faster calculations)
    state = particle_df["state"].values
    init_state = state[1]  # second value
    t = particle_df.t.values
    T = t[-1] - t[0]
    ind = state == 0.5
    ind[0] = False
    ind[-1] = False
    t_stick_unstick = t[ind]
    K = len(t_stick_unstick)
    dt_for_jac = 1.

    logP_k = np.log(kEventsProb(K, init_state, T, T_stick, T_unstick))
    logL_traj = specificTrajectoryLikelihood(trajData, init_state, t_stick_unstick, dt_for_jac, T_stick, T_unstick, D,
                                             A)
    logL_total = logP_k + logL_traj
    return logL_total


def experimentLikelihood(df, T_stick, T_unstick, D, A, Parallelized=False, equal_particle_weight=False):
    t_start = time.time()

    particles_array = df.particle.unique()
    grouped = df.groupby("particle")
    particle_df = [grouped.get_group(i) for i in particles_array]

    if equal_particle_weight:
        weights = np.double(grouped.apply(len).values) ** (-1)  # inverse traj length
    else:
        weights = np.ones(len(particles_array))
    weights = weights / sum(weights)  # normalize

    if Parallelized:
        args_iterable = []
        logp_experiment = 0
        with ProcessPoolExecutor(max_workers=1) as executor:
            for j, logp_particles in enumerate(executor.map(labeledParticleLikelihood, particle_df, repeat(T_stick),
                                                            repeat(T_unstick), repeat(D), repeat(A))):
                logp_experiment += logp_particles * weights[j]
                print(j)
        # for i in particles_array:
        #     args_iterable.append(np.asarray((grouped.get_group(i), T_stick, T_unstick, D, A),dtype=object))
        # # with Pool() as pool:
        # with Pool(1) as pool:
        #     logp_particles = pool.starmap(particleLikelihood, args_iterable)
    else:
        logp_particles = np.zeros(len(particles_array), dtype=object)
        for i, particle_id in enumerate(particles_array):
            logp_particles[i] = labeledParticleLikelihood(grouped.get_group(particle_id), T_stick, T_unstick, D, A)
        logp_experiment = np.sum(logp_particles * weights)

    try:
        print("Estimated experiment likelihood of " + "{:.2f}".format(logp_experiment) + " in " + "{:.2f}".format(
            time.time() - t_start) + " sec")
    except:
        print("Estimated experiment likelihood derivatives in " + "{:.2f}".format(
            time.time() - t_start) + " sec")

    if np.isnan(logp_experiment):
        print("logp_experiment is nan - printing parameter values:")
        print("T_st = ", T_stick, "\nT_un = ", T_unstick, "\nD = ", D, "\nA = ", A)

    return logp_experiment


def basicMaxLikelihood(df, x0, cons_arr=[0.1, 0.1, 1., 0.0001], scaling_factor=[10., 10., 10., 100.], usePar=False,
                       maxIter=100, savefile_extra="", fixedArguments=[False, False, False, False]):
    # T_stick | T_unstick | D | A
    # scaling factor > 1 is for increasing constraints and decreasing value in function (too small vars that make the constraint too small and the optimizer crosses it)
    cons_arr = np.array(cons_arr)
    scaling_factor = np.array(scaling_factor)

    x0_orig = x0
    x0 = x0 * scaling_factor
    cons_arr = cons_arr * scaling_factor

    fixedArgs = np.where(np.asarray(fixedArguments))

    L = lambda x: -1 * experimentLikelihood(df, x[0] / scaling_factor[0], x[1] / scaling_factor[1],
                                            x[2] / scaling_factor[2], x[3] / scaling_factor[3], Parallelized=usePar)

    cons = ({'type': 'ineq',
             'fun': lambda x: x[0] - cons_arr[0]},
            {'type': 'ineq',
             'fun': lambda x: x[1] - cons_arr[1]},
            {'type': 'ineq',
             'fun': lambda x: x[2] - cons_arr[2]},
            {'type': 'ineq',
             'fun': lambda x: x[3] - cons_arr[3]})

    # autograd tensorflow...jax tfprobability
    print("Beginning Minimization:")
    t_start = time.time()
    # minimum = scipy.optimize.minimize(L, x0, constraints=cons, options={'disp': True, 'maxiter': maxIter})
    if usePar:
        minimum = scipy.optimize.minimize(L, x0=x0, constraints=cons)  # , maxIter=maxIter)
    else:
        minimum = scipy.optimize.minimize(L, x0=x0, constraints=cons,
                                          )#jac=jacobian(L)),  # maxIter=maxIter)  # ,  hess=hessian(L))
    filename = "likelihood_data_opt" + "_" + savefile_extra
    filepath = "./files/" + filename
    backupBeforeSave(filepath)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    np.save(filepath, [minimum, x0_orig, df])
    print("Saved results in " + filepath)
    print("Total optimization time of " + "{:.2f}".format(time.time() - t_start) + " sec")
    print(minimum)
    return minimum


def gridLikelihood(df, T_stick_arr, T_unstick_arr, D_arr, A_arr):
    args_iterable = itertools.product(T_stick_arr, T_unstick_arr, D_arr, A_arr)
    args_list = list(args_iterable)
    args_iterable = itertools.product([df], T_stick_arr, T_unstick_arr, D_arr, A_arr)
    # parallelized = False  # parallelization is in the experiment likelihood
    # if parallelized:
    #     with Pool() as pool:
    #         L_tensor = pool.starmap(experimentLikelihood, args_iterable)
    # else:
    t_start = time.time()

    list_args_iterable = list(args_iterable)
    L_tensor = np.zeros(len(list_args_iterable))
    for i, args in enumerate(list_args_iterable):
        L_tensor[i] = experimentLikelihood(args[0], args[1], args[2], args[3], args[4])
    L_tensor = list(L_tensor)

    print("Elapsed time = " + "{:.2f}".format(time.time() - t_start) + "sec")

    # save here
    filename = "likelihood_data_grid"
    filepath = "./files/" + filename
    backupBeforeSave(filepath)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    np.save(filepath, [args_list, L_tensor, df])

    max_likelihood_params = args_list[np.argmax(L_tensor)]
    T_stick = max_likelihood_params[0]
    T_unstick = max_likelihood_params[1]
    D = max_likelihood_params[2]
    A = max_likelihood_params[3]

    L_sorted = np.sort(L_tensor)
    print("Next highest log-Likelihoods: " + str(L_sorted[-4:-1] - L_sorted[-1]))
    print("T_st = ", T_stick, "\nT_un = ", T_unstick, "\nD = ", D, "\nA = ", A)
    return L_tensor


def extract_trajData_from_df(df):
    t = df.t.values[:]
    if np.isnan(df.dt.values[0]):
        dt = df.dt.values[1:]  # delta t per step (first value is nan)
        dr2 = df.dr2.values[1:]  # delta r squared per step
    else:
        dt = df.dt.values  # delta t per step (first value is nan)
        dr2 = df.dr2.values  # delta r squared per step
    x = df.x.values
    y = df.y.values
    trajData = [t, dt, dr2, x, y]  # this eliminates the need for df
    return trajData


def segmentTraj(len_df, baseLen=64.):
    # Divide long df to smaller df, return indices
    segSize = np.floor(len_df / np.max([np.round(len_df / baseLen), 1.01]))
    start_ind = np.arange(0, len_df - 1, step=segSize)
    if (len(start_ind) > 1) & (len_df - start_ind[-1] < baseLen / 2 + 1):
        start_ind = start_ind[:-1]
    end_ind = 0 * start_ind
    end_ind[0:-1] = start_ind[1:] - 1
    end_ind[-1] = len_df - 1
    segment_mat = np.vstack([start_ind, end_ind]).transpose()
    return segment_mat.astype(int)


# @primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    maxVal = x.max()
    exp_arr = np.zeros(x.shape, dtype=object)
    for i, xval in enumerate(x):
        exp_arr[i] = np.exp(xval - maxVal)
    LSE = maxVal + np.log(exp_arr.sum())
    return LSE

# def logsumexp_vjp(ans, x):
#     x_shape = x.shape
#     return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))
#
#
# defvjp(logsumexp, logsumexp_vjp)
