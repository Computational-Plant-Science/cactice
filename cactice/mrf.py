import collections
import random
from itertools import product
from typing import List, Tuple, Callable, Dict, OrderedDict

import numpy as np
from numpy.random import choice, RandomState

import cactice.stats as stats

# bond interaction signature
# params:
#  - grid
#  - location A
#  - location B
# returns: bond energy
Interaction = Callable[[np.ndarray, Tuple[int, int], Tuple[int, int]], float]


def bond_energies(
        grid: np.ndarray,
        interaction: Interaction,) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
    """
    Computes cardinal-direction bonds and bond energies for the given grid and interaction function.

    TODO: allow neighbor specification (cardinal, diagonal, complete)

    :param grid: The grid
    :param interaction: The interaction function
    :return: A dictionary mapping bonds to bond energies
    """

    bonds = dict()
    h, w = grid.shape

    # horizontal interactions
    for i, j in product(range(w - 1), range(h)):
        e = interaction(grid, (i, j), (i + 1, j))
        bonds[((i, j), (i + 1, j))] = e

    # vertical interactions
    for i, j in product(range(w), range(h - 1)):
        e = interaction(grid, (i, j), (i, j + 1))
        bonds[((i, j), (i, j + 1))] = e

    return bonds


def H(grid: np.ndarray, interaction: Interaction, J: float = 1.0) -> float:
    """
    Computes the Hamiltonian (energy function) of the given grid

    :param grid: The grid
    :param interaction: The bond interaction
    :param J: The multiplier
    :return: The energy
    """
    bonds = bond_energies(grid, interaction)
    e = sum(bonds.values())
    return J * e


def neighborhood_H(
        grid: np.ndarray,
        cell: Tuple[int, int],
        interaction: Interaction,
        J: float = 1.0) -> float:
    """
    Computes the Hamiltonian (energy function) of the given neighborhood.

    :param grid: The grid
    :param cell: The cell around which the neighborhood is centered
    :param interaction: The bond interaction
    :param J: The multiplier
    :return: The energy
    """

    h, w = grid.shape
    i, j = cell
    e = 0

    if i > 0: e += interaction(grid, (i - 1, j), (i, j))
    if i < w - 1: e += interaction(grid, (i, j), (i + 1, j))
    if j > 0: e += interaction(grid, (i, j - 1), (i, j))
    if j < h - 1: e += interaction(grid, (i, j), (i, j + 1))

    return J * e


class MRF:
    def __init__(
            self,
            interaction: Interaction,
            J: float = 1.0,
            iterations: int = 250,
            threshold: float = 0.01,
            seed: int = 42):
        """
        Create a Markov random field model.

        :param interaction: The bond interaction
        :param J: The multiplier
        :param iterations: The number of iterations of the Metropolis algorithm to run
        :param threshold: The probability of accepting a detrimental update
        :param seed: The random seed
        """

        self.__random_state = RandomState(seed)
        self.__interaction: Interaction = interaction
        self.__J: float = J
        self.__iterations: int = iterations
        self.__threshold: float = threshold
        self.__train: List[np.ndarray] = []
        self.__class_distribution = Dict[str, float]

    def fit(self, grids: List[np.ndarray]):
        """
        Fit the model to the given grids (precompute neighborhoods and probability distribution).
        """

        self.__train = grids
        self.__class_distribution = stats.classes(grids)

    def predict(self, grids: List[np.ndarray] = None):
        """
        Predict missing cells on the training grids or on the given grids (if provided).
        To generate entirely novel grids conditioned on the training set, provide a list of empty (zero-valued) arrays.
        This method uses the Metropolis algorithm to minimize the bond energy (roughly, "surprise") on the predicted grids, conditioned on the training grids' bond distribution.

        :param grids: The grids to predict on (the training set will be used otherwise)
        :return: The grids with missing values filled in.
        """

        # initialize grids to predict on
        grid_predictions = [np.copy(grid) for grid in (grids if grids is not None else self.__train)]

        for gi, grid in enumerate(grid_predictions):
            # find cells to predict (missing locations)
            h, w = grid.shape
            rows = range(0, h)
            cols = range(0, w)
            missing: List[Tuple[int, int]] = [(i, j) for i in rows for j in cols if grid[i, j] == 0]

            pred = np.copy(grid)
            accepted = 0
            rejected = 0
            total_energies = []
            average_energies = []
            updates: OrderedDict[Tuple[int, int], int] = collections.OrderedDict()

            # randomly initialize missing cells
            for i, j in missing: pred[i, j] = choice(self.__class_distribution, 1)[0]

            # proceed while we haven't reached the cutoff point
            while accepted < self.__iterations and rejected < self.__iterations:
                # make a copy of the grid
                pcpy = np.copy(pred)

                # pick random missing location
                i, j = missing[random.randint(0, len(missing) - 1)]

                # make random selection from class distribution
                cell = choice(self.__class_distribution, 1)[0]
                pcpy[i, j] = cell

                # compute the energy pre- and post-update and calculate difference
                energy_old = neighborhood_H(pred, (i, j), self.__interaction)
                energy_new = neighborhood_H(pcpy, (i, j), self.__interaction)
                difference = energy_new - energy_old

                # compute the total and average energy corresponding to the new configuration
                energy_ttl = H(np.vectorize(lambda x: float(x))(pcpy), self.__interaction)
                energy_avg = energy_ttl / len([p for p in np.ravel(grid) if p != 0])

                # if we lowered the energy (or random chance of detriment if we didn't), accept the update
                if difference < 0 or self.__random_state.uniform() > (1 - self.__threshold):
                    accepted += 1
                    total_energies.append(energy_ttl)
                    average_energies.append(energy_avg)
                    updates[(i, j)] = cell

                    # update predicted grid
                    pred[i, j] = cell
                else:
                    rejected += 1
                    if len(total_energies) > 0:
                        total_energies.append(total_energies[-1])
                        average_energies.append(average_energies[-1])

            # save predicted grid
            grid_predictions[gi] = pred

        return grid_predictions
