import numpy as np
from utils import isnotebook
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from config import MAX_PROCESSORS_LIMIT, MAX_AVAILABLE_STATES_TO_KEEP, FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO
from model import Model, State


def viterbi_algorithm(X_arr, T_stick: float, T_unstick: float, D: float, A: float, dt: float, log_process=False):
    """
    Given a particle's series of observed positions X_arr, and model parameters, find the most likely sequence of hidden
    states S and X_tether.

    The algorithm closely follows the Wikipedia implementation, however there are key points to note:
    Our model has a continuum of possible States, specifically the hidden state X_tether derived from S and the
    position X_arr. The algorithm, however, requires a fixed state space of K states. For our model, at the time n
    of the trajectory, there are n+2 possible states: free, or still stuck at X[n], or stuck at X[n-1], and so on until
    still stuck at X[0]. So overall for the trajectory of length N, there are N+1 possible states (N+1 = (N-1)+2).

    We define a hidden state index (i and j) as follows: i=0 is, and i=k is stuck at X[k-1], so 0<=i<=K=N+1.
    Each of this indices, along with a time tag n, can be converted to the appropriate model State by adding the
    particle position X_arr[n] to the hidden state. Then the model can be used to calculate the likelihood of transition
    between two States.

    We mention the Viterbi algorithm's strength is discarding paths on the go as we choose the best path to
    get to a destination hidden state and discard the rest. However in this model, only the free state i=0 can be
    accessed from multiple states, so we only discard the paths leading to it.

    Finally, because the number of states grows linearly with the time along the trajectory, we introduce two additional
    filters to discard more paths and ease the computation, unrelated to the Viterbi scheme itself:

        1.  If the particle's position at time n is quite far from its position at time m<n, then the likelihood that
            it is still stuck at X[m] is very low, so at time n we discard the hidden state i=m+1 corresponding to still
            being stuck there. The global parameter FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO controls what is considered
            far (how many A of a squared distance between particles is enough for discarding).

        2.  If the particle is stuck for a long time, the above filter will not work and the number of states will grow.
            However, it is not truly necessary to consider all options of sticking and unsticking when confined to a
            small region. Therefore at each time step, we keep only the free state and the MAX_AVAILABLE_STATES_TO_KEEP-1 most likely
            stuck states (so that overall at any time step there are no more than MAX_AVAILABLE_STATES_TO_KEEP allowed states).

    Args:
        X_arr: Nx2 array of particle positions (x,y)
        T_stick: model parameter
        T_unstick: model parameter
        D: model parameter
        A: model parameter
        dt: time step between samples [Todo later: replace with t_arr to allow for uneven sampling]
        log_process: print algorithm progress.


    Returns:
       S_arr (int array of length N): states S of the most likely trajectory (i.e. the Viterbi path).
       X_tether_arr (Nx2 float array): tether points X_tether of the most likely trajectory
       viterbi_path_log_likelihood (float): the log-likelihood of the most likely trajectory

    """
    # 1. Initialization
    model = Model(T_stick, T_unstick, D, A, dt)
    N = len(X_arr)  # trajectory length
    K = N + 1  # total number of possible hidden states

    # Construct State objects:
    from_state = State(0, [0., 0.], [0., 0.])
    to_state = State(0, [0., 0.], [0., 0.])

    def L(n, i, j):
        """
        Calculate the log-likelihood of transition from hidden state i at time n-1 to hidden state j at time n
        """

        # First values in from_state and to_state
        ind2state(n - 1, i, from_state, X_arr)
        ind2state(n, j, to_state, X_arr)

        # Then calculate by the model
        return model.transition_log_likelihood(from_state, to_state)

    # Log-likelihood matrix L_mat[i,n] is the log-likelihood of the most likely path that
    # reaches state i at time n (n time points so this is the sum of n-1 step contributions + initial condition)
    L_mat = np.zeros([K, N], dtype=float)

    # State matrix S_mat[i,n] is the most likely state at time n-1 which leads to state i (at time n).
    # Assign -1 to everything as this is not a valid state.
    S_mat = np.zeros([K, N], dtype=int) - 1

    # More parameters
    squared_distance_discard_threshold = FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO * A  # For the first filter
    isNotebook = isnotebook()  # For logging

    # Initialize: equal prior for stuck or free at time 0 (leave L[:,0]=0)
    # all unreached end destinations are by default with 0 probability, and only if we reach them then their probability
    # is to be considered (some states won't be reached due to the paths discarding).
    L_mat[:, - 1] = -np.inf

    hidden_states_list = np.arange(K, dtype=int)
    valid_hidden_states_mask = np.zeros(K, dtype=bool)
    valid_hidden_states_mask[0] = True

    # 2. Loop along the trajectory
    for n in range(1, N):  # time - we follow the Wikipedia Viterbi entry with indices j,i,k

        # Filter 1: forget the states of staying stuck somewhere very far away from current positon X[n]
        valid_hidden_states_mask[1:] *= np.sum((X_arr - X_arr[n]) ** 2, axis=1) <= squared_distance_discard_threshold

        # Filter 2: keep only the MAX_AVAILABLE_STATES_TO_KEEP-2 most likely stuck states

        # if np.sum(valid_hidden_states_mask) > MAX_AVAILABLE_STATES_TO_KEEP - 1:
        #     valid_hidden_states_mask[
        #         1 + np.argsort(L_mat[hidden_states_list[valid_hidden_states_mask][1:], n - 1])[
        #             :-(MAX_AVAILABLE_STATES_TO_KEEP - 2)]] = False

        # Allow for the recently stuck at n-1 state to be valid. The free state is always valid.
        valid_hidden_states_mask[n] = True

        available_source_hidden_states = hidden_states_list[
            valid_hidden_states_mask]  # these are the available SOURCE states
        # j is DESTINATION state - its allowed states are the available SOURCE states + state n+1
        for j in available_source_hidden_states:
            if j == 0:  # only the free state has multiple paths leading to it (more than one source state)
                # Choose the best path to j from all available states i - this is the Viterbi algorithm!
                logL_paths_to_j = [L_mat[i, n - 1] + L(n, i, j) for i in available_source_hidden_states]
                ind_max = np.argmax(logL_paths_to_j)
                S_mat[j, n] = available_source_hidden_states[ind_max]
                L_mat[j, n] = logL_paths_to_j[ind_max]
                # print("States: {}".format(available_source_hidden_states))
                # print("ind max = {}, S = {}".format(ind_max,available_source_hidden_states[ind_max]))

            else:
                # for 0<j<n+1, can only get to j from j (stuck in the same place)
                # as long as this i is still valid (very far tether points are forgotten)
                S_mat[j, n] = j
                L_mat[j, n] = L_mat[j, n - 1] + L(n, j, j)
        # In addition consider the sticking from free (i=0) to stuck at location j (state j=n+1)
        S_mat[n + 1, n] = 0
        L_mat[n + 1, n] = L_mat[0, n - 1] + L(n, 0, n + 1)
        # Some logging:
        if log_process:
            if n % int(np.ceil(N / 10)) == 0:
                if isNotebook:
                    print("Processed {}% of the trajectory".format(int(np.floor(100 * n / N))), end="\r")
                else:
                    print("Processed {}% of the trajectory".format(int(np.floor(100 * n / N))))

    # 3. Matrices are filled - now we backprop to the start to find the path itself:
    # We find the best end point and then trace it back to obtain the full path
    viterbi_path_final_state = np.argmax(L_mat[:, - 1])
    viterbi_path_log_likelihood = L_mat[viterbi_path_final_state, - 1]
    viterbi_path = np.zeros(N, dtype=int)
    viterbi_path[N - 1] = viterbi_path_final_state
    # Now follow along state matrix S_mat
    for n in reversed(range(1, N)):
        viterbi_path[n - 1] = S_mat[viterbi_path[n], n]

    # Now convert this from the indices i to S and X_tether:
    # (for efficiency we don't use ind2state for this part)
    S_arr = np.zeros(N, dtype=int)
    X_tether_arr = np.zeros([N, 2], dtype=float)

    for n in range(N):
        i = viterbi_path[n]  # state
        if i == 0:
            S_arr[n] = 0
            X_tether_arr[n, :] = np.nan
        else:
            S_arr[n] = 1
            X_tether_arr[n, :] = X_arr[i - 1, :]

    if log_process:
        if isNotebook:
            print("")
        print("Done with Viterbi Algorithm!")
    return S_arr, X_tether_arr, viterbi_path_log_likelihood


def ind2state(n: int, i: int, state: State, X_arr):
    """
    Convert a discrete hidden state index i at time n to a State from the model's state space. Assigns value in input
    state object to avoid constructing each time. See hidden state index in the Viterbi algorithm documentation.

    Args:
        n - time step
        i - hidden state index
        state - State object in which to store the state properties (instead of constructing a new one each time)
        X_arr - array of positions (from which the position and tether points are obtained)

    Doesn't return anything, instead assigning the values in the input state.
    """

    state.X = X_arr[n]
    if i == 0:
        state.S = 0
        # don't care about X_tether if free
    else:
        state.S = 1
        state.X_tether = X_arr[i - 1]


def extract_X_arr_list_from_df(df):
    return [particle_df[["x", "y"]].values for _, particle_df in df.groupby("particle")]


def extract_dt_list_from_df(df):
    return [particle_df["dt"].mean() for _, particle_df in df.groupby("particle")]


def multiple_trajectories_likelihood(X_arr_list, dt_list, T_stick: float, T_unstick: float, D: float, A: float,
                                     is_parallel=False):
    """
    Normalized likelihood of observed trajectories of multiple particles given the model parameters. We make the
    assumption that the full trajectories, including the hidden states, are just the Viterbi paths (instead of
    integrating over all possible hidden trajectories which is the "correct" way to tell the likelihood of the observed
    trajectories). Furthermore we normalize each particle's trajectory likelihood by its trajectory length.

    Args:
        X_arr_list (Nx1 list of matrices): list of the observed positions of all particles - each element of the list is
        an [Mx2] ndarray of the observed positions, where M is the specific particle's trajectory length.
        dt_list (Nx1 ndarray): list of the time interval for the sampling for each particle.
        T_stick : model paramaeter
        T_unstick: model parameter
        D: model parameter
        A: model parameter
        is_parallel: parallelized computing for all particles (should be True unless the optimization function calling
        this is already parallelized)

    Returns:
        The sum of each of the particles' Viterbi paths likelihood, divided by their trajectory lengths
        (equal weight per particle).

    """
    N_particles = len(dt_list)
    L_arr = np.zeros(N_particles)  # log likelihood array
    length_arr = np.asarray([len(X_arr) for X_arr in X_arr_list])  # trajectory length array

    if is_parallel:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for n, output in enumerate(executor.map(viterbi_algorithm, X_arr_list, repeat(T_stick),
                                                    repeat(T_unstick), repeat(D), repeat(A), dt_list)):
                L_arr[n] = output[2]
    else:
        for n in range(N_particles):
            _, _, L = viterbi_algorithm(X_arr_list[n], T_stick, T_unstick, D, A, dt_list[n], log_process=False)
            L_arr[n] = L

    return np.sum(L_arr / length_arr) / N_particles
