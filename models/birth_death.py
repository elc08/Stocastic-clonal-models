# =============================================================================
# Created By  : Eric Latorre
# Created Date: 2021-08-16
# =============================================================================
"""Birth and death model for clonality where SCs acquire mutations following
a Gaussian process.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np

def bd(t, lamb=1.3, mu=1.3, tmax=90, from_zero=False):
    ''' Stochastic B-D model of a clone.
    Parameters:
    - t: float. Start time
    - lamb: float. Birth rate (events/ year).
    - mu: float. Death (events/ year).
    - tmax: int age of individuals for stopping time
    - from_zero: Bool. Choose wether checkpoints are recorded from clone birth
                       or from time 0.
    Returns:
    - checkpoints: Array. Recorded population of cells every year.
    '''


    if from_zero is False:
        # compute first recorded time_point
        t0 = int(np.ceil(t))
        # create checkpoints array
        checkpoints = np.zeros((2, int(tmax)-t0+1))
        checkpoints[0] = np.linspace(t0, int(tmax),
                                     int(tmax)-t0+1)
        # initialise cells
        clone_size = 1
        # Append initial population if start time is a checkpoint.
        if t == t0:
            checkpoints[1, 0] = clone_size

    if from_zero is True:
        t0 = int(0)
        # create checkpoints array starting at 0
        checkpoints = np.zeros((2, int(tmax)+1))
        checkpoints[0] = np.linspace(t0, int(tmax),
                                     int(tmax)+1)

        # initialise cells
        clone_size = 1
        if int(t) == t:
            checkpoints[1, t] = clone_size

    # model loop
    while t <= tmax:
        # save current time in previous_t
        previous_t = t

        # update birth and death rates
        lamb_x = lamb * clone_size
        mu_x = mu * clone_size

        #  Timestep to next event
        rate = lamb_x + mu_x
        t += -np.log(np.random.uniform(0, 1)) / rate

        # Check if any timepoints occurred between t_previous and t.
        # If so append x for all such timepoints
        checkpoints[1, int(previous_t)-t0+1:int(t)-t0+1] = clone_size

        # Decision of which event happened during time-step
        urv = np.random.uniform(0, 1)
        if urv >= lamb_x / rate:
            clone_size -= 1
            # if division leads to clonal death exit simulation
            if clone_size == 0:
                # If clone dies set t > tmax to exit simulation
                t = tmax + 1
        else:
            clone_size += 1

    return checkpoints
