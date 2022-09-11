import numpy as np
from numpy import random
from model_utils import GenerationMode
from collections import namedtuple
import numba


#THIS HAS TO BE CONSISTENT WITH FUNCTION pack_model_params!
Params = namedtuple('Params',
                    ['T_stick', 'T_unstick', 'D', 'A', 'dt',
                     'MSD', 'log_pi_MSD', 'inertia_factor', 'modified_2A', 'log_pi_modified_2A',
                     'log_stay_free', 'log_stick', 'log_unstick', 'log_stay_stuck',
                     'd_A_tilde_d_D', 'd_A_tilde_d_A'])


def pack_model_params(T_stick: float, T_unstick: float, D: float, A: float, dt: float):
    """
    Pack a namedtuple containing all the model parameters, and also all the derived parameters used for the calculation
    (e.g. calculate the log of some parameters here ones, instead of computing log each time)
    This is namedtuple instead of dict for compatibility with numba
    (see https://stackoverflow.com/questions/46003172/replacement-of-dict-type-for-numba-as-parameters-of-a-python-function)
    """
    d = 2  # dimension
    MSD = 4 * D * dt  # comment: this is not really MSD in dimensions other than 2 - true MSD is (d/2) * 4*D*dt
    r = 1. / T_stick + 1. / T_unstick  # combined rate

    inertia_factor = np.exp(-D * dt / A)
    modified_2A = 2 * A * (1 - inertia_factor ** 2)

    log_pi_MSD = 0.5 * d * np.log(np.pi * MSD)
    log_pi_modified_2A = 0.5 * d * np.log(np.pi * modified_2A)

    phi = np.exp(-r * dt)
    log_stay_free = np.log((T_stick + T_unstick * phi) / (T_stick + T_unstick))
    log_stay_stuck = np.log((T_unstick + T_stick * phi) / (T_stick + T_unstick))
    log_stick = np.log((T_unstick * (1 - phi)) / (T_stick + T_unstick))
    log_unstick = np.log((T_stick * (1 - phi)) / (T_stick + T_unstick))

    d_A_tilde_d_D = 2 * dt * (1 - modified_2A / (2 * A))
    d_A_tilde_d_A = (1 - MSD / (2 * A)) * modified_2A / (2 * A) - MSD / (2 * A)

    model_params = Params(T_stick, T_unstick, D, A, dt,
                          MSD, log_pi_MSD, inertia_factor, modified_2A, log_pi_modified_2A,
                          log_stay_free, log_stick, log_unstick, log_stay_stuck,
                          d_A_tilde_d_D, d_A_tilde_d_A)

    return model_params


@numba.jit(nopython=True)
def model_transition_log_probability(S_prev, S_curr, X_prev, X_curr, X_tether_prev, model_params):
    """
    Calculate the log of probability of transition from one state to another, when they are separated by time dt, using
    the model parameters. In essence this incorporates all the model information.

    Args: [M is 1 or number of transitions for vectorized implementation thanks to broadcast)]
        S_prev (M array of int): 0 for free, 1 for stuck, of the state the transition is FROM
        S_curr (M array of int): 0 for free, 1 for stuck, of the state the transition is TO
        X_prev (Mx2 float): position (x,y) of the state the transition is FROM
        X_curr (Mx2 float): position (x,y) of the state the transition is TO
        X_tether_prev (Mx2 float): tether position (x,y) of the state the transition is FROM
        model_params: this includes all the model parameters, the time step dt, and some products and logs of them
        (comment: don't need X_tether_curr)

        self: Model class - this provides the 4 model parameters, and optionally dt and P
        from_state: the state from which there is a transition to the destination state
        to_state: the destination state
        dt (float): optional argument of the delta t between the states, if not given then take from self

    Returns:
        L (float): log of the probability of the transition. Could be -np.inf for zero-probability transitions.
    """

    # Now allows for vectorized implementations

    # bool array
    prev_free = (S_prev == 0)
    prev_stuck = np.logical_not(prev_free)
    curr_free = (S_curr == 0)
    curr_stuck = np.logical_not(curr_free)

    # Need to do the addition here because of broadcasting in axis 1
    dX_free = (X_curr - X_prev)
    # But we want to sum over axis 1 which is impossible if it's a 2x1 array... fix this shit for clean vectorization
    if len(dX_free.shape) == 1:
        summation_ax_free = 0
    else:
        summation_ax_free = 1
    spatial_free = - np.sum(dX_free ** 2, axis=summation_ax_free) / model_params.MSD - model_params.log_pi_MSD

    dX_stuck = (X_curr - X_tether_prev) - model_params.inertia_factor * (X_prev - X_tether_prev)
    if len(dX_stuck.shape) == 1:
        summation_ax_stuck = 0
    else:
        summation_ax_stuck = 1
    spatial_stuck = - np.sum((dX_stuck) ** 2, axis=summation_ax_stuck) / (
        model_params.modified_2A) - model_params.log_pi_modified_2A

    spatial = prev_free * spatial_free + prev_stuck * spatial_stuck
    temporal = prev_free * (curr_free * model_params.log_stay_free + curr_stuck * model_params.log_stick) + \
               prev_stuck * (curr_free * model_params.log_unstick + curr_stuck * model_params.log_stay_stuck)
    L = spatial + temporal

    return L


def model_generate_trajectories(N_steps: int, N_particle: int, init_S, model_params: dict,
                                generation_mode=GenerationMode.DONT_FORCE):
    """
    Generate a series of States drawn from the distribution corresponding to the model. This has a vectorized
    implementation for generating trajectories of multiple particles. All particles have the same trajectory length.

    Args:
        N_steps: duration of trajectory (in steps) for each of the particles
        N_particles: how many trajectories to generate. Note: all trajectories start at X=[0,0].
        init_S: initial S for each of the particles: 0 is free, 1 is stuck, None is random for each particle (50%).
        generation_mode: FORCE_FREE and FORCE_STUCK make all the particles free or stuck all the time. DONT_FORCE
        allows for transitions according to the model.
        model_params: dict with parameters of the model

    Returns:
        states_arr (N_particle x N_steps int ndarray): states for each particle at each time step.
        X_arr (N_particle x N_steps x 2 float ndarray): positions (x,y) for each particle at each time step.
        X_tether_arr (N_particle x N_steps x 2 float ndarray): tether point (x,y) for each particle at each step.
        (Comment: effectively this is a N_particle x N_step State matrix.)

    """
    reduce_tethering_range = 0.  # 1e-10 # MOVE THIS TO ARGUMENTS OR SMNTHG

    if generation_mode == GenerationMode.FORCE_FREE:
        init_S = 0
    elif generation_mode == GenerationMode.FORCE_STUCK:
        init_S = 1

    init_state_arr = np.zeros(N_particle)
    if init_S == 1:
        init_state_arr += 1
    elif init_S is None:
        init_state_arr = np.random.randint(2, size=N_particle).astype(int)

    states_arr = np.zeros([N_particle, N_steps], dtype=int)
    X_arr = np.zeros([N_particle, N_steps, 2])
    X_tether_arr = np.zeros([N_particle, N_steps, 2])
    states_arr[:, 0] = init_state_arr

    P = np.zeros([2, 2])
    if generation_mode == GenerationMode.DONT_FORCE:
        # note: this is valid only when dt<<T_stick,T_unstick
        P[0, 0] = np.exp(model_params.log_stay_free)
        P[0, 1] = np.exp(model_params.log_stick)
        P[1, 0] = np.exp(model_params.log_unstick)
        P[1, 1] = np.exp(model_params.log_stay_stuck)
    if generation_mode == GenerationMode.FORCE_FREE:
        P[:, 0] = 0.
        P[:, 1] = 1.
    elif generation_mode == GenerationMode.FORCE_STUCK:
        P[:, 0] = 1.
        P[:, 1] = 0.

    # Stream of 2D gaussian RV with variance 1
    gaussian_Stream = random.default_rng().normal(loc=0.0, scale=1, size=[N_particle, N_steps, 2])

    # This is for when a particle sticks and another random sample is needed for the tether point
    extra_gaussian_Stream = random.default_rng().normal(loc=0.0, scale=1,
                                                        size=[N_particle, N_steps, 2])

    uniform_Stream = random.default_rng().random(size=[N_particle, N_steps])

    # for clarity X_tether is initialized only for stuck
    X_tether_arr[np.where(init_state_arr == 1.), 0, :] = reduce_tethering_range * np.sqrt(
        0.5 * model_params.modified_2A) * gaussian_Stream[np.where(init_state_arr == 1.), 0, :]

    for n in range(1, N_steps):
        free_inds = np.where(states_arr[:, n - 1] == 0.)[0]
        stuck_inds = np.where(states_arr[:, n - 1] == 1.)[0]

        # Free particles diffuse
        X_arr[free_inds, n, :] = X_arr[free_inds, n - 1, :] + \
                                 np.sqrt(0.5 * model_params.MSD) * gaussian_Stream[free_inds, n, :]
        X_tether_arr[free_inds, n, :] = np.nan
        # Stuck particles wiggle

        X_arr[stuck_inds, n, :] = X_tether_arr[stuck_inds, n - 1, :] * (1 - model_params.inertia_factor) + \
                                  X_arr[stuck_inds, n - 1, :] * model_params.inertia_factor + \
                                  np.sqrt(0.5 * model_params.modified_2A) * gaussian_Stream[stuck_inds, n, :]

        # Tether point continues UNLESS going to stick
        X_tether_arr[:, n, :] = X_tether_arr[:, n - 1, :]

        # Stick or unstick:
        sticking_inds = free_inds[np.where(uniform_Stream[free_inds, n] > P[0, 0])[0]]
        staying_free_inds = np.setdiff1d(free_inds, sticking_inds)
        unsticking_inds = stuck_inds[np.where(uniform_Stream[stuck_inds, n] > P[1, 1])[0]]
        staying_stuck_inds = np.setdiff1d(stuck_inds, unsticking_inds)

        states_arr[np.union1d(unsticking_inds, staying_free_inds), n] = 0.
        states_arr[np.union1d(sticking_inds, staying_stuck_inds), n] = 1.

        # Sticking particles tether to a point
        X_tether_arr[sticking_inds, n, :] = X_arr[sticking_inds, n, :] + \
                                            reduce_tethering_range * \
                                            np.sqrt(0.5 * model_params.modified_2A) * \
                                            extra_gaussian_Stream[sticking_inds, n, :]

    return states_arr, X_arr, X_tether_arr


def model_trajectory_log_probability(S_arr, X_arr, model_params: dict):
    """
    Calculate the log of probability of one particle's trajectory (including hidden states)

    Args:
        S_arr (Nx1 int): list of states - 0 is free, 1 is stuck
        X_arr (Nx2 float): list of particle positions
        model_params: this includes all the model parameters, the time step dt, and some products and logs of them

    Returns:
        L (float): log of the probability of the trajectory (sum of the probabilities of each transition)
    """
    N = len(S_arr)
    L = 0.
    X_tether = X_arr[0]  # placeholder unless S[0]!=0 and then we assume the particle is stuck at X[0].
    for n in range(1, N):
        L += model_transition_log_probability(S_arr[n - 1], S_arr[n], X_arr[n - 1], X_arr[n], X_tether, model_params)
        # Update the tether point
        # (it's ok that is after the likelihood calculation because X_tether matters only if S_arr[n-1]!=0)
        if (S_arr[n - 1] == 0) and (S_arr[n] != 0):
            X_tether = X_arr[n]

    return L


def model_get_optimal_parameters(dt, S_arr, X_arr, X_tether_arr=None):
    """
    Calculate the most likely model parameters given a full trajectory

    Args:
        dt (float): time interval
        S_arr (Nx1 int): list of states - 0 is free, 1 is stuck
        X_arr (Nx2 float): list of particle positions
        X_tether_arr (Nx2 float): list of particle tether positions. optional argument (can be derived from the others)

    Returns:
        4x1 array of model parameters [T_stick,T_unstick,D,A]
    """
    if np.prod(S_arr == 0) == 1:
        T_stick_est = np.inf
        T_unstick_est = np.nan
        D_est = np.mean(np.sum((X_arr[1:, :] - X_arr[:-1, :]) ** 2, axis=1)) / (4 * dt)
        A_est = np.nan
    elif np.prod(S_arr != 0) == 1:
        T_stick_est = np.inf
        T_unstick_est = np.nan
        D_est = np.nan
        A_est = np.nanmean(np.sum((X_arr[1:, :] - X_tether_arr[:-1, :]) ** 2, axis=1)) / (2)
    else:
        N = len(S_arr)
        if X_tether_arr is None:
            X_tether_arr = np.zeros([N, 2])
            X_tether_arr[0, :] = X_arr[0]
            for n in range(1, N):
                if (S_arr[n - 1] == 0) and (S_arr[n] != 0):
                    X_tether_arr[n, :] = X_arr[n]
                else:
                    X_tether_arr[n, :] = X_tether_arr[n - 1, :]

        mask_free_to_free = (S_arr[:-1] == 0) * (S_arr[1:] == 0)
        mask_free_to_stuck = (S_arr[:-1] == 0) * (S_arr[1:] != 0)
        mask_stuck_to_free = (S_arr[:-1] != 0) * (S_arr[1:] == 0)
        mask_stuck_to_stuck = (S_arr[:-1] != 0) * (S_arr[1:] != 0)

        mask_free = mask_free_to_free + mask_free_to_stuck
        mask_stuck = mask_stuck_to_free + mask_stuck_to_stuck

        # wrapped with if because of pesky warnings
        if np.sum(mask_free_to_stuck) > 0:
            T_stick_est = (np.sum(mask_free_to_free) / np.sum(mask_free_to_stuck) + 1) * dt
        else:
            T_stick_est = np.inf
        if np.sum(mask_stuck_to_free) > 0:
            T_unstick_est = (np.sum(mask_stuck_to_stuck) / np.sum(mask_stuck_to_free) + 1) * dt
        else:
            T_unstick_est = np.inf
        D_est = np.sum((mask_free * np.sum((X_arr[1:, :] - X_arr[:-1, :]) ** 2, axis=1))) / (4 * dt) / np.sum(mask_free)
        A_est = np.nansum((mask_stuck * np.sum((X_arr[1:, :] - X_tether_arr[:-1, :]) ** 2, axis=1))) / 2 / np.sum(
            mask_stuck)

    return np.asarray([T_stick_est, T_unstick_est, D_est, A_est])
