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
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# Enable the table_schema option in pandas,
# data-explorer makes this snippet available with the `dx` prefix:
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None


def birth(clone):
    """Birth event.

    Birth event creates a new cell with identic mutations to a
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
    """Death event.

    Death event takes away a randomly selected cell.
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

def clonal_evolution(tmax=80, rate_d=1.3, rate_s=1.3, rate_m=0.1):

    # start a clone with an initial mutation
    # each element of clone_mutation_list corresponds to a living cell
    clone = np.ones((1, 1))
    clone_size = 1
    clone_history = []
    clone_history.append(clone.sum(axis=0))
    t = 0

    while t < tmax:
        total_cells = clone.shape[0]

        propensities = np.array([rate_m * total_cells,
                                 rate_s * total_cells,
                                 rate_d * total_cells])

        total_propensity = propensities.sum()

        # gillespie algorithm

        # Time to next event
        random_gillespie = np.random.uniform()
        tau = -np.log(random_gillespie) / total_propensity

        # Check if a checkpoint is crossed before the next event
        previous_t = t
        t = t + tau

        # Check if checkpoint ws crossed
        if int(previous_t) != int(t):

            # update clone history in checkpoints
            for i in range(int(previous_t)+1, int(t)+1):
                clone_history.append(clone.sum(axis=0))

            for year in range(int(t)):
                result = np.zeros(clone_history[int(t)].shape)
                result[:clone_history[year].shape[0]] = clone_history[year]
                clone_history[year] = result

        prob_propensities = propensities/total_propensity
        i = np.random.choice(3, p=prob_propensities)

        if i == 0:
            clone = mutation(clone)

        elif i == 1:
            clone = birth(clone)
            clone_size += 1
        elif i == 2:
            clone = death(clone)
            clone_size -= 1
            if clone_size == 0:
                break

    clone_history = np.array(clone_history)
    full_clone_history = np.zeros((int(tmax)+1, clone_history.shape[1]))
    full_clone_history[:clone_history.shape[0]] = clone_history

    return full_clone_history


init_cells = 1_000
clone_track = []
for i in tqdm(range(init_cells)):
    clone_track.append(clonal_evolution(tmax=10))

total_cells = np.array([clone[:, 0] for clone in clone_track])
total_cells = total_cells.sum(axis=0)
total_cells
fig_total_cells = px.line(title='Clone size',
                          x=list(range(total_cells.shape[0])),
                          y=total_cells)
fig_total_cells.show(renderer='svg')


fig = go.Figure()
for clone in clone_track:
    x = list(range(clone.shape[0]))
    for i in range(clone.shape[1]):
        fig.add_trace(
            go.Scatter(x=x, y=clone[:,i]))
fig.show(renderer="svg")
