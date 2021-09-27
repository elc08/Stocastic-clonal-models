# =============================================================================
# Created By  : Eric Latorre
# Created Date: 2021-09-27
# =============================================================================
"""Birth and death model for clonality where SCs acquire mutations following
a Gaussian process.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np


# =============================================================================
# Auxiliary functions
# =============================================================================

def competitive_bd(t, tmax=80, lamb=1.3, mu=1.3, init=np.ones(500),
                   capacity = 1_000,
                   feedback_size=False,
                   quiescence=False,
                   fit=False):
    ''' Stochastic B-D model of a clone
        t0: Float. initiating time.
        tmax: Int. Simulation stopping time.
        lamb: Float. birth rate / year
        mu: Float. death rate / year
        init: Array. Initial clonal distribution.
        capacity: Float. Neutral level of the system.
        feedback_size: Float.
        quiescence: Bool. If True, clones with 1 SC cannot die but rather stay
                          in a quiescent state.
        fit: Bool. If True, one initial clone escapes feedback regulation.
    '''

    # set a seed for reproducibility
    # the seed shouldn't be applied for the test scenario of
    # all mutations starting at 0
    #np.random.seed(int(t*10_000))

    t0 = int(t)

    checkpoints = np.zeros((init.shape[0], tmax-t0+1, 2))
    checkpoints[:,:,0] = np.linspace(t0, tmax, tmax-t0+1)
    checkpoints[:,0,1] = init
    x = init

    cell_history = np.e*np.ones(init.shape[0])
    history_checkpoints = np.zeros((init.shape[0], tmax-t0+1, 2))
    history_checkpoints[:,:,0] = np.linspace(t0, tmax, tmax-t0+1)
    history_checkpoints[:,0,1] = cell_history

    # Create a for loop to track progress of time
    for counter in tqdm(range(1,tmax+1)):
        if sum(x == 0) == x.shape[0]:
            break
        # model loop
        while t <= counter:
            # save current time in previous_t
            previous_t = t

            if feedback_size is False:
                feedback_regulation = 1
            else:
                # propensity with feedback regulation
                feedback_regulation = 1./(feedback_size*cell_history)
                if fit is True:
                    feedback_regulation[0] = 1

            # update birth and death rates
            lamb_propensity = x*(2*lamb/capacity)*(capacity-sum(x))*feedback_regulation

            # update death propensity
            if feedback_size is False:
                # propensity without feedback regulation
                mu_propensity = x*mu

            else:
                if quiescence is False:
                    mu_propensity = x*mu*feedback_regulation
                else:
                    # compute propensity where x!=1
                    mu_propensity = np.where(x == 1, 0, x)*mu*feedback_regulation

            rate = np.sum(lamb_propensity) + np.sum(mu_propensity)

            #  Timestep to next event
            t += -np.log(np.random.uniform(0, 1)) / rate

            # If t > tmax append last x to all timepoints
            # between previous_t and tmax
            # and exit loop
            if t >= tmax:
                checkpoints[:,int(previous_t)-t0+1:,:].T[1] = x
                history_checkpoints[:,int(previous_t)-t0+1:,:].T[1] = cell_history
                break

            # Check if any timepoints occurred between t_previous and t.
            # If so append x for all such timepoints
            checkpoints.T[1,int(previous_t)-t0+1: int(t)-t0+1] = x
            history_checkpoints.T[1,int(previous_t)-t0+1: int(t)-t0+1] = cell_history


            # Decision of which event happened during time-step
            urv = np.random.uniform(0, 1)
            if urv >= np.sum(lamb_propensity) / rate:
                # Death
                prob_rates = mu_propensity/np.sum(mu_propensity)
                i = np.random.choice(range(len(x)), p=prob_rates)
                x[i] -= 1
            else:
                prob_rates = lamb_propensity/np.sum(lamb_propensity)
                i = np.random.choice(range(len(x)), p=prob_rates)
                x[i] += 1
                cell_history[i] += 1

            # Exit the loop if no cells remain alive
            if sum(x == 0) == x.shape[0]:
                break

    return checkpoints, history_checkpoints
