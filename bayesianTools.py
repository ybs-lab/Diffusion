import numpy as np
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import scipy

from config import MAX_PROCESSORS_LIMIT, MAX_AVAILABLE_STATES_TO_KEEP, FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO
from model import model_transition_log_probability, pack_model_params, model_get_optimal_parameters
import numba


@numba.jit(nopython=True)
def viterbi_algorithm(X_arr, model_params, do_backprop=True, ):
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
        do_backprop: if True then return the Viterbi path (via backprop) and its likelihood. If False, just return
                     the best path's likelihood without getting the path itself (use False for optimization)


    Returns:
       S_arr (int array of length N): states S of the most likely trajectory (i.e. the Viterbi path).
       X_tether_arr (Nx2 float array): tether points X_tether of the most likely trajectory
       viterbi_path_log_likelihood (float): the log-likelihood of the most likely trajectory

    """
    # 1. Initialization
    N = len(X_arr)  # trajectory length
    K = N + 1  # total number of possible hidden states

    # Log-likelihood matrix L_mat[i,n] is the log-likelihood of the most likely path that
    # reaches state i at time n (n time points so this is the sum of n-1 step contributions + initial condition)
    L_mat = np.zeros((K, N), dtype=np.float32)

    # State matrix S_mat[i,n] is the most likely state at time n-1 which leads to state i (at time n).
    # Assign -1 to everything as this is not a valid state.
    S_mat = np.zeros((K, N), dtype=np.int32) - 1

    # More parameters
    squared_distance_discard_threshold = FORGET_FAR_TETHER_POINTS_THRESHOLD_RATIO * model_params.A  # For the first filter
    # Initialize: equal prior for stuck or free at time 0 (leave L[:,0]=0)
    # all unreached end destinations are by default with 0 probability, and only if we reach them then their probability
    # is to be considered (some states won't be reached due to the paths discarding).
    L_mat[:, - 1] = -np.inf

    hidden_states_list = np.arange(K, dtype=np.int32)
    valid_hidden_states_mask = np.zeros(K, dtype=np.bool_)  # must have np.bool_ and not np.bool for numba to work!
    valid_hidden_states_mask[0] = True

    # 2. Loop along the trajectory
    for n in range(1, N):  # time - we follow the Wikipedia Viterbi entry with indices j,i,k

        # Filter 1: forget the states of staying stuck somewhere very far away from current positon X[n]
        valid_hidden_states_mask[1:] *= np.sum((X_arr - X_arr[n]) ** 2, axis=1) <= squared_distance_discard_threshold

        # Filter 2: keep only the MAX_AVAILABLE_STATES_TO_KEEP-2 most likely stuck states
        if np.sum(valid_hidden_states_mask) > MAX_AVAILABLE_STATES_TO_KEEP - 1:
            nonfree_valid_hidden_states = hidden_states_list[valid_hidden_states_mask][1:]
            valid_hidden_states_mask[
                nonfree_valid_hidden_states[
                    np.argsort(L_mat[nonfree_valid_hidden_states, n - 1])
                ][: -(MAX_AVAILABLE_STATES_TO_KEEP - 2)]] = False

        # Allow for the recently stuck at n-1 state to be valid. The free state is always valid.
        valid_hidden_states_mask[n] = True

        available_source_hidden_states = hidden_states_list[
            valid_hidden_states_mask]  # these are the available SOURCE states
        # j is DESTINATION state - its allowed states are the available SOURCE states + state n+1

        for j in available_source_hidden_states:
            if j == 0:  # only the free state has multiple paths leading to it (more than one source state)
                # Choose the best path to j from all available states i - this is the Viterbi algorithm!
                logL_paths_to_j = L_mat[available_source_hidden_states, n - 1] + \
                                  model_transition_log_probability(S_prev=available_source_hidden_states, S_curr=j,
                                                                   X_prev=X_arr[n - 1], X_curr=X_arr[n],
                                                                   X_tether_prev=X_arr[
                                                                       available_source_hidden_states - 1],
                                                                   model_params=model_params)
                #               transition_log_probability_viterbi(
                # model_params, X_arr, n, available_source_hidden_states, j)
                ind_max = np.argmax(logL_paths_to_j)
                S_mat[j, n] = available_source_hidden_states[ind_max]
                L_mat[j, n] = logL_paths_to_j[ind_max]

            else:
                # for 0<j<n+1, can only get to j from j (stuck in the same place)
                # as long as this i is still valid (very far tether points are forgotten)
                S_mat[j, n] = j
                L_mat[j, n] = L_mat[j, n - 1] + \
                              model_transition_log_probability(S_prev=j, S_curr=j,
                                                               X_prev=X_arr[n - 1], X_curr=X_arr[n],
                                                               X_tether_prev=X_arr[
                                                                   j - 1],
                                                               model_params=model_params)
                # transition_log_probability_viterbi(model_params, X_arr, n, j, j)
        # In addition consider the sticking from free (i=0) to stuck at location j (state j=n+1)
        S_mat[n + 1, n] = 0
        L_mat[n + 1, n] = L_mat[0, n - 1] + \
                          model_transition_log_probability(S_prev=0, S_curr=n + 1,
                                                           X_prev=X_arr[n - 1], X_curr=X_arr[n],
                                                           X_tether_prev=X_arr[n - 1],  # this has no meaning here
                                                           model_params=model_params)

        # transition_log_probability_viterbi(model_params, X_arr, n, 0, n + 1)
    # 3. Matrices are filled - now we backprop to the start to find the path itself:
    # We find the best end point and then trace it back to obtain the full path
    viterbi_path_final_state = np.argmax(L_mat[:, - 1])
    viterbi_path_log_likelihood = L_mat[viterbi_path_final_state, - 1]

    if do_backprop:
        # Find the best trajectory and not just the best path's likelihood
        S_arr = np.zeros(N, dtype=np.int32)  # can reduce to bool
        X_tether_arr = np.zeros((N, 2), dtype=np.float32)
        viterbi_path = np.zeros(N, dtype=np.int32)
        viterbi_path[N - 1] = viterbi_path_final_state
        # Now follow along state matrix S_mat
        for n in np.flip(np.arange(1, N, dtype=np.int32)):
            viterbi_path[n - 1] = S_mat[viterbi_path[n], n]

        # Now convert this from the indices i to S and X_tether:
        # (for efficiency we don't use ind2state for this part)

        for n in range(N):
            i = viterbi_path[n]  # state
            if i == 0:
                S_arr[n] = 0
                X_tether_arr[n, :] = np.nan
            else:
                S_arr[n] = 1
                X_tether_arr[n, :] = X_arr[i - 1, :]
    else:
        S_arr = None
        X_tether_arr = None

    return S_arr, X_tether_arr, viterbi_path_log_likelihood


def extract_X_arr_list_from_df(df):
    return np.asarray([particle_df[["x", "y"]].values for _, particle_df in df.groupby("particle")])


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

    model_params = pack_model_params(T_stick, T_unstick, D, A, np.mean(dt_list))  # TODO: FIX THIS!!!

    if is_parallel:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
            for n, output in enumerate(executor.map(viterbi_algorithm, X_arr_list, repeat(model_params), repeat(False)
                                                    )):
                L_arr[n] = output[2]
    else:
        for n in range(N_particles):
            _, _, L = viterbi_algorithm(X_arr_list[n], model_params,
                                        do_backprop=False)
            L_arr[n] = L

    return np.sum(L_arr / length_arr) / N_particles


def em_viterbi_optimization(X_arr_list, dt_list, T_stick: float, T_unstick: float, D: float, A: float,
                            is_parallel=False, max_iterations=100,verbose=False):
    """
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
        max_iterations: maximal number of iterations (usually there is convergence after 5-6 iters and the loop stops)

    Returns:
        params_by_iter (N_iter x 4 ndarray): the 4 model parameters estimated per iteration
        L_by_iter (N_iter x 1 ndarray): the likelihood of the parameters per iteration
        S_arr_list (list of len N_particles): list that contains ndarray with array of the hidden variable S for each particle
        X_tether_arr_list (list of len N_particles): list that contains ndarray with matrix of the hidden variable X_tether for each particle
    """

    N_particles = len(dt_list)
    L_by_iter = np.zeros(max_iterations)
    params_by_iter = np.zeros([max_iterations + 1, 4])
    L_arr = np.zeros(N_particles)  # log likelihood array
    S_arr_list = np.empty(N_particles, dtype=object)
    X_tether_arr_list = np.empty(N_particles, dtype=object)
    length_arr = np.asarray([len(X_arr) for X_arr in X_arr_list])  # trajectory length array

    dt = np.mean(dt_list)  # TODO: FIX THIS!!
    initial_parameters = np.asarray([T_stick, T_unstick, D, A])  # for documentation
    params_by_iter[0, :] = initial_parameters
    prev_parameters = np.asarray([T_stick, T_unstick, D, A])  # for documentation
    model_params = pack_model_params(T_stick, T_unstick, D, A, dt)

    if verbose:
        print("Initial parameters are [{:.2e}, {:.2e}, {:.2e}, {:.2e}]".format(
            initial_parameters[0], initial_parameters[1], initial_parameters[2], initial_parameters[3]
        ))

    for iter in range(max_iterations):
        # E step - find trajectories
        if verbose:
            print("Starting iteration {} - E step".format(iter))
        if is_parallel:
            with ProcessPoolExecutor(max_workers=MAX_PROCESSORS_LIMIT) as executor:
                for n, output in enumerate(
                        executor.map(viterbi_algorithm, X_arr_list, repeat(model_params), repeat(True)
                                     )):
                    S_arr_list[n] = output[0]
                    X_tether_arr_list[n] = output[1]
                    L_arr[n] = output[2]
        else:
            for n in range(N_particles):
                output = viterbi_algorithm(X_arr_list[n], model_params,
                                           do_backprop=True)
                S_arr_list[n] = output[0]
                X_tether_arr_list[n] = output[1]
                L_arr[n] = output[2]

        L_by_iter[iter] = np.sum(L_arr)
        # M step - find the parameters
        params_mat = np.zeros([N_particles, 4])
        for n in range(N_particles):
            params_mat[n, :] = model_get_optimal_parameters(
                dt_list[n], S_arr_list[n], X_arr_list[n], X_tether_arr_list[n])
        params_mat[~np.isfinite(params_mat)] = np.nan
        params_est = np.nanmean(params_mat, axis=0)
        params_est[~np.isfinite(params_est)] = prev_parameters[~np.isfinite(params_est)]
        prev_parameters = params_est.copy()
        T_stick, T_unstick, D, A = params_est
        model_params = pack_model_params(T_stick, T_unstick, D, A, dt)
        params_by_iter[iter + 1, :] = params_est
        if verbose:
            print("Done with iteration {} - M step".format(iter))
            print("Current parameter estimates are [{:.2e}, {:.2e}, {:.2e}, {:.2e}]".format(
                params_est[0],params_est[1],params_est[2],params_est[3]
            ))

        if (iter > 0) and (np.abs(L_by_iter[iter] - L_by_iter[iter - 1]) / (
                np.abs(L_by_iter[iter]) + np.abs(L_by_iter[iter - 1]))) < 1e-10:
            break

    params_by_iter = params_by_iter[:(iter + 1), :]
    L_by_iter = L_by_iter[:iter]

    return params_by_iter, L_by_iter, S_arr_list, X_tether_arr_list


def generate_Q_matrix(N, model_params):
    diag_arr = np.zeros(N + 1)
    diag_arr[1:] = (1 - model_params.dt / model_params.T_unstick) / (np.pi * model_params.modified_2A)
    diag_arr[0] = (1 - model_params.dt / model_params.T_stick) / (np.pi * model_params.MSD)
    Q = scipy.sparse.diags(diag_arr, 0).tocsr()
    Q[1:, 0] = (model_params.dt / model_params.T_unstick) / (np.pi * model_params.modified_2A)
    Q.eliminate_zeros()
    return Q


def generate_P_matrix(X_arr, n, model_params, Q, N, return_grads=True):
    # need n <= N-2
    Q = Q.copy()
    Q[0, n + 2] = (model_params.dt / model_params.T_stick) / (np.pi * model_params.MSD)

    diag_arr = np.zeros(N + 1)
    diag_arr[0] = -np.sum((X_arr[n + 1] - X_arr[n]) ** 2) / model_params.MSD
    diag_arr[1:(n + 2)] = -np.sum((X_arr[n + 1] - X_arr[:(n + 1)]) ** 2, axis=1) / model_params.modified_2A
    exp_diag_arr = diag_arr.copy()
    exp_diag_arr[:(n + 2)] = np.exp(diag_arr[:(n + 2)])

    P = scipy.sparse.diags(exp_diag_arr, 0) @ Q

    # max_P = P.max()
    # P = np.round(P / max_P, 10) * P

    P.eliminate_zeros()

    # #should i do this???????????
    # for m in range(P.shape[0]):
    #     P[m,:]/= P[m,:].sum()

    if not return_grads:
        return P

    for_spatial_grads = scipy.sparse.diags(1 + diag_arr, 0) @ P
    gradients_arr = [P.copy() if i <= 1 else for_spatial_grads.copy() for i in range(4)]

    dt = model_params.dt

    T_stick = model_params.T_stick
    # T_stick
    gradients_arr[0][1:, :] = 0
    gradients_arr[0][0, 1:] /= -T_stick
    gradients_arr[0][0, 0] /= (T_stick * (T_stick / dt - 1))

    T_unstick = model_params.T_unstick
    # T_unstick
    gradients_arr[1][0, :] = 0
    gradients_arr[1][1:, :] /= (T_unstick * (T_unstick / dt - 1))
    gradients_arr[1][1:, 0] /= -T_unstick

    D = model_params.D
    modified_A = model_params.modified_2A / 2

    # D
    gradients_arr[2][0, :] *= (-1 / D)
    gradients_arr[2][1:, :] *= -(model_params.d_A_tilde_d_D / modified_A)

    # A
    gradients_arr[3][0, :] *= 0
    gradients_arr[3][1:, :] *= -(model_params.d_A_tilde_d_A / modified_A)

    for i in range(4):
        gradients_arr[i].eliminate_zeros()
    return P, gradients_arr


def matrix_gd_scheme(X_arr, model_params):
    N = len(X_arr)
    Q = generate_Q_matrix(N, model_params)

    # pi_mat = scipy.sparse.coo_matrix(np.zeros([N, N + 1]), (N, N + 1)).tocsr()
    grads_row_mat = [scipy.sparse.coo_matrix(np.zeros([N - 1, N + 1]), (N - 1, N + 1)).tocsr()
                     for m in range(4)]
    grads_arr = np.zeros([N - 1, 4])

    # INITIAL CONDITION
    pi = np.zeros([1, N + 1])
    pi[:2] = 0.5

    # END VECTOR
    end_vector = np.ones([N + 1, 1])

    log_L = 0.

    for n in range(N - 1):
        P, grads = generate_P_matrix(X_arr, n, model_params, Q, N)

        # if n == 2:
        #     print(P)
        #     return 0, 0
        for m in range(4):
            grads_row_mat[m][n, :] = pi @ grads[m]
        pi = pi @ P

        pi_sum = np.sum(pi)
        pi /= np.sum(pi)
        log_L += np.log(pi_sum)

    L = (pi @ end_vector)[0][0] * np.exp(log_L)
    for n in np.flip(np.arange(N - 1, dtype=np.int32)):
        for m in range(4):
            grads_arr[n, m] = grads_row_mat[m][n, :] @ end_vector
        P = generate_P_matrix(X_arr, n, model_params, Q, N, return_grads=False)
        end_vector = P @ end_vector

    gradients = np.sum(grads_arr, axis=0)
    return L, gradients
