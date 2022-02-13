import random
from typing import List

import numpy as np

import cactice.stats as stats
from cactice.neighbors import Neighbors, get_neighborhood


class RNS:
    def __init__(self, neighbors: Neighbors = Neighbors.CARDINAL, layers: int = 1):
        """
        Create a random neighbor selection model.
        This model assigns to each cell by simply selecting at random from its neighbors.
        If no neighbors are known, a value is randomly selected from the observed class distribution.

        :param neighbors: The cells to consider part of the neighborhood.
        :param layers: The width of the neighborhood
        """
        self.__neighbors = neighbors
        self.__layers = layers

    def predict(self, grids: List[np.ndarray] = None):
        # initialize grids to predict on
        grid_predictions = [np.copy(grid) for grid in grids]

        # compute class distribution
        class_distribution = stats.classes(grids)

        for gi, grid in enumerate(grid_predictions):
            # find cells to predict (missing locations)
            missing = [(i, j) for i in range(0, grid.shape[0]) for j in range(0, grid.shape[1]) if grid[i, j] == 0]

            # predict cells one by one
            for i, j in missing:
                # get the missing location's neighbors
                neighborhood = get_neighborhood(grid, i, j, self.__neighbors, self.__layers)
                neighbors = list(neighborhood.values())[1:]  # first element is central cell
                any_neighbors = len(neighborhood.keys()) > 1

                # predict cell value by making a random selection from its neighbors, if any
                # or if none, choosing randomly according to the observed class distribution
                cell_pred = random.choice(neighbors) if any_neighbors else np.random.choice(
                    list(class_distribution.keys()),
                    list(class_distribution.values()))

                # set the cell in the corresponding grid
                grid_predictions[gi][i, j] = cell_pred

        return grid_predictions
