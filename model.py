import numpy as np
from numpy import random
from model_utils import GenerationMode


class State:
    """
    The state of a particle in our model.

    Attributes:
        S (int): 0 for free, 1 for stuck
        X (2x1 array-like): the particle's position
        X_tether (2x1 array-like): the particle's tether point
    """

    def __init__(self, S: int, X, X_tether):
        self.S = S
        self.X = X
        self.X_tether = X_tether


class Model:
    """
    Our Markov model for the particle dynamics - alternating between free diffusion and tethering.

    The model comprises of 4 parameters: T_stick,T_unstick,D,A. This class has methods for calculating the transition
    log-likelihood between two states (class State) and for generating trajectories according to the model.

    Attributes:
        T_stick (float): mean sticking time (seconds)
        T_unstick (float): mean unsticking time (seconds)
        D (float): diffusion coefficient in the free state (um^2/second)
        A (float): mean area of the potential well in the stuck state (um^2)
        dt (float): time between two samples (seconds) - this serves as a default for the class methods
        P (2x2 array-like): transition matrix between free and stuck states (no units)
            [Comment: P is an attribute just to save repeated computations of the matrix]

    """

    def __init__(self, T_stick: float, T_unstick: float, D: float, A: float, dt: float):
        """
        Set attributes and also calculate the P matrix with the input dt, to save further calculations.
        """
        self.T_stick = T_stick
        self.T_unstick = T_unstick
        self.D = D
        self.A = A
        self.dt = dt
        self.P = self.generate_CTMC_Matrix()

    def transition_log_likelihood(self, from_state: State, to_state: State, dt=None, enforce_same_tether=False):
        """
        Calculate the likelihood of transition from one state to another, when they are separated by time dt, using
        the model parameters (from the class attributes). In essence this incorporates all the model information.

        Args:
            self: Model class - this provides the 4 model parameters, and optionally dt and P
            from_state: the state from which there is a transition to the destination state
            to_state: the destination state
            dt (float): optional argument of the delta t between the states, if not given then take from self
            enforce_same_tether (bool): return -inf if there is a transition of stuck to stuck with different tethers.
                (if this restriction is enforced in an outside algorithm, no need to spend computation on checking).


        Returns:
            L (float): log-likelihood of the transition. Could be -np.inf for zero-probability transitions.
        """

        # First enforce the tether to stay the same unless this is a sticking event or the particle is not stuck
        # Doing this enforcement slows the function, and this enforcement can be ensured at the iterative Viterbi algorithm
        # [don't link states tethered to Y0 with a state tethered to Y1]
        if enforce_same_tether and to_state.S == 1 and from_state.S == 1 and not np.all(
                np.isclose(from_state.X_tether, to_state.X_tether)):
            return -np.inf

        if dt is None:
            dt = self.dt
            P = self.P
        else:
            P = self.generate_CTMC_Matrix(dt)
        # Calculate as normal from here
        # Temporal part (state transitions)
        L = np.log(P[from_state.S, to_state.S])
        # Spatial part
        if from_state.S == 0:  # free
            MSD = 4 * self.D * dt  # expected MSD in time dt
            L += -(np.log(np.pi * MSD) + np.sum(to_state.X - from_state.X) ** 2 / MSD)
        else:
            L += -(np.log(np.pi * self.A) + np.sum(to_state.X - from_state.X_tether) ** 2 / self.A)
        return L

    def generate_CTMC_Matrix(self, dt=None):
        if dt is None:
            dt = self.dt
        P = np.zeros([2, 2])
        r = 1. / self.T_stick + 1. / self.T_unstick  # combined rate
        phi = np.exp(-r * dt)
        P[0, 0] = self.T_stick + self.T_unstick * phi
        P[0, 1] = self.T_unstick * (1 - phi)
        P[1, 0] = self.T_stick * (1 - phi)
        P[1, 1] = self.T_unstick + self.T_stick * phi
        P /= (self.T_stick + self.T_unstick)
        return P

    def generate_trajectories(self, N_steps: int, N_particle: int, init_S, generation_mode=GenerationMode.DONT_FORCE):
        """
        Generate a series of States drawn from the distribution corresponding to the model. This has a vectorized
        implementation for generating trajectories of multiple particles. All particles have the same trajectory length.

        Args:
            self: Model class - this provides the 4 model parameters AND the time step dt
            N_steps: duration of trajectory (in steps) for each of the particles
            N_particles: how many trajectories to generate. Note: all trajectories start at X=[0,0].
            init_S: initial S for each of the particles: 0 is free, 1 is stuck, None is random for each particle (50%).
            generation_mode: FORCE_FREE and FORCE_STUCK make all the particles free or stuck all the time. DONT_FORCE
            allows for transitions according to the model.

        Returns:
            states_arr (N_particle x N_steps int ndarray): states for each particle at each time step.
            X_arr (N_particle x N_steps x 2 float ndarray): positions (x,y) for each particle at each time step.
            X_tether_arr (N_particle x N_steps x 2 float ndarray): tether point (x,y) for each particle at each step.
            (Comment: effectively this is a N_particle x N_step State matrix.)

        """
        reduce_tethering_range = 0.  # 1e-10 # MOVE THIS TO ARGUMENTS OR SMNTHG

        if generation_mode == GenerationMode.FORCE_FREE:
            init_S = 0.
        elif generation_mode == GenerationMode.FORCE_STUCK:
            init_S = 1.

        init_state_arr = np.zeros(N_particle)
        if init_S == 1:
            init_state_arr += 1
        elif init_S is None:
            init_state_arr = np.random.randint(2, size=N_particle).astype(float)

        states_arr = np.zeros([N_particle, N_steps])
        X_arr = np.zeros([N_particle, N_steps, 2])
        X_tether_arr = np.zeros([N_particle, N_steps, 2])
        states_arr[:, 0] = init_state_arr

        P = self.P.copy()
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
        X_tether_arr[np.where(init_state_arr == 1.), 0, :] = reduce_tethering_range*np.sqrt(self.A) * gaussian_Stream[
                                                                               np.where(init_state_arr == 1.), 0,
                                                                               :]

        # note: this is valid only when dt<<T_stick,T_unstick

        for n in range(1, N_steps):
            free_inds = np.where(states_arr[:, n - 1] == 0.)[0]
            stuck_inds = np.where(states_arr[:, n - 1] == 1.)[0]

            # Free particles diffuse
            X_arr[free_inds, n, :] = X_arr[free_inds, n - 1, :] + np.sqrt(4 * self.D * self.dt) * gaussian_Stream[
                                                                                                  free_inds,
                                                                                                  n, :]
            X_tether_arr[free_inds,n,:]=np.nan
            # Stuck particles wiggle
            X_arr[stuck_inds, n, :] = X_tether_arr[stuck_inds, n - 1, :] + np.sqrt(self.A) * gaussian_Stream[stuck_inds,
                                                                                             n, :]

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
                                                reduce_tethering_range * np.sqrt(self.A) * extra_gaussian_Stream[
                                                                                           sticking_inds, n, :]
        return states_arr, X_arr, X_tether_arr
