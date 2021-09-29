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


# =============================================================================
# Auxiliary functions
# =============================================================================
def birth(clone):
    """Birth event creates a new cell with identic mutations to a
    randomly selected cell.

    Parameters:
    clone: Array. Clone.

    Returns:
    new_clone: Array. Updated array of the system.
    """
    np.random.shuffle(clone)
    new_clone = np.vstack((clone, clone[0]))

    return new_clone


def death(clone):
    """Death event takes away a randomly selected cell.

    Parameters:
    clone: Array. Clone.

    Returns:
    new_clone: Array. Updated array of the system.
    """
    np.random.shuffle(clone)
    new_clone = clone[:-1, :]

    return new_clone


def mutation(clone):
    """ Mutation event: Creates a new column of binary mutation event
    then randomly selects a cell to be mutated.

    Parameters:
    clone: Array. Clone.

    Returns:
    new_clone: Array. Updated array of the system"""

    # append new column of zeros to all cells
    new_clone = np.zeros((clone.shape[0], clone.shape[1]+1))
    new_clone[:, :-1] = clone

    # select cell being mutated
    np.random.shuffle(new_clone)
    new_clone[0, -1] = 1

    return new_clone


def pad_history(element, shape):
    if element is None:
        return np.zeros(shape)
    new = np.pad(element, (0, shape - element.shape[0]), 'constant')
    return new


# =============================================================================
# Stochastic simulation
# =============================================================================
def clonal_evolution(tmax=90, death_rate=1.3, birth_rate=1.3,
                     mutation_rate=0.064):
    """ Stocastic simulation of a SC clone accumulating mutations with a
    constant rate over time.

    Mutation rate in adulthood corresponds to 16 mutation/cell/year.
    If we only observe 100 genes of a total of 25_000, this leads to a mutation
    rate of 0.064.

    Parameters
    ----------
    tmax: float. Length of simulation in years.
    birth_rate: float. Birth rate /cell/year.
    death_rate: float. Death rate /cell/year.
    mutation_rate: float. Mutation rate /cell/year.

    Returns
    ----------
    clone_history: array. Each row corresponds to a SC and
                          each column to the binary presence of a mutation.
    """

    # start a clone with an initial mutation
    # each row of clone corresponds to a living cell
    # each column of clone tracks a mutation for all cells
    # The presence of each mutation is encoded binary
    clone = np.ones((1, 1))

    # create a record of the state of the amount of cells
    # for each mutation at each year
    clone_history = np.empty(int(tmax)+1, dtype=object)
    clone_history[0] = clone.sum(axis=0)

    # initialise time
    t = 0

    # loop in time
    while t < tmax:
        # total_cells can be extracted as the number of rows in clone
        total_cells = clone.shape[0]

        # compute propensities
        propensities = np.array([mutation_rate * total_cells,
                                 birth_rate * total_cells,
                                 death_rate * total_cells])

        total_propensity = propensities.sum()

        # Gillespie algorithm
        # Time to next event
        random_gillespie = np.random.uniform()
        tau = -np.log(random_gillespie) / total_propensity

        # Check if a checkpoint is crossed before the next event
        previous_t = t
        t = t + tau

        # Check if checkpoint was crossed and update history
        if int(previous_t) != int(t):
            # compute how many timepoints are crossed
            broadcast_shape = len(clone_history[int(previous_t)+1:int(t)+1])
            # Append copies of the clonal state of the system to history
            clone_history[int(previous_t)+1:int(t)+1] = (
                [clone.sum(axis=0)]*broadcast_shape)

        # Check which event happened
        prob_propensities = propensities/total_propensity
        i = np.random.choice(3, p=prob_propensities)

        if i == 0:
            clone = mutation(clone)

        elif i == 1:
            clone = birth(clone)

        elif i == 2:
            clone = death(clone)
            if clone.shape[0] == 0:
                break

    # Pad with zeros all recorded timepoints to create a full history array
    total_mutations = max([len(i) for i in clone_history if i is not None])

    clone_history = np.array([pad_history(i, total_mutations)
                              for i in clone_history], dtype=object)

    return clone_history


def delete_ghosts(clone):
    """ Delete mutations that were born and died out within checkpoints.

    Parameters
    ----------
    - clone: Array. Clone steming from initial SC.

    Returns
    ----------
    -filter_clone: Array. Filter clonal array wihtout empty mutations.
    """

    # Locate columns of empty mutations
    idx = np.argwhere(np.all(clone[..., :] == 0, axis=0))
    # delete empty columns
    filter_clone = np.delete(clone, idx, axis=1)
    return filter_clone


def filter_clone(clone):
    """ Delete rows after clonal death.

    Parameters
    ----------
    - clone: Array. Clone steming from initial SC.

    Returns
    ----------
    -filter_clone: Array. Filter clonal array wihtout empty rows.
    """

    # Find extinction time using clone size
    extinction_time = (clone[:, 0] == 0).argmax()

    if extinction_time > 0:
        # if there is an extinction time, filter all rows after extinction.
        filtered_clone = clone[:extinction_time+1, :]
    else:
        filtered_clone = clone
    return filtered_clone
