from typing import List, Dict, Tuple
from itertools import islice
from collections import Counter

import numpy as np

# TODO: more distance function options?
from cactice.neighbors import get_neighborhood, Neighbors
from cactice.distance import hamming_distance


class KNN:
    def __init__(self, k: int = 10, neighbors: Neighbors = Neighbors.CARDINAL, layers: int = 1):
        """
        Create a K-nearest neighbors model.

        :param neighbors: Which adjacent cells to consider neighbors.
        :param layers: How many layers of adjacent cells to consider neighbors.
        """
        self.__k = k
        self.__neighbors = neighbors
        self.__layers = layers

        # store neighborhoods as a list of grids,
        # each grid a dictionary mapping absolute cell coordinates (i, j) to neighborhoods,
        # each neighborhood a dictionary mapping relative cell coordinates (i, j) to values
        self.__neighborhoods: List[Dict[Tuple[int, int], Dict[Tuple[int, int], int]]] = []

    def fit(self, grids: List[np.ndarray]):
        """
        Fit the model to the given grids (precompute neighborhoods).
        """

        # set the training set
        self.__train = grids

        # for each grid...
        for grid in grids:
            # compute neighborhood at each cell
            neighborhoods = {(i, j): get_neighborhood(grid, i, j, self.__neighbors, self.__layers) for i in range(0, grid.shape[0]) for j in range(0, grid.shape[1])}

            self.__neighborhoods.append(neighborhoods)

    def predict(self, grids: List[np.ndarray] = None):
        """
        Predict missing cells on the training grids or on the given grids (if provided).
        To generate entirely novel grids conditioned on the training set, provide a list of empty (zero-valued) arrays.

        :param grids: The grids to predict on (if none are provided the training set will be used).
        :return: The predicted grids.
        """

        # initialize grids to predict on
        grid_predictions = [np.copy(grid) for grid in (grids if grids is not None else self.__train)]

        # flatten all training grids' neighborhoods into one list
        neighborhoods = [h for hs in [list(nhds.values()) for nhds in self.__neighborhoods] for h in hs]

        for gi, grid in enumerate(grid_predictions):
            # find cells to predict (missing locations)
            missing = [(i, j) for i in range(0, grid.shape[0]) for j in range(0, grid.shape[1]) if grid[i, j] == 0]

            # predict cells one by one
            for i, j in missing:
                # get the missing location's neighbors
                neighborhood = get_neighborhood(grid, i, j, self.__neighbors, self.__layers)

                # compute distance from this neighborhood to every training neighborhood
                distances = {nh[(0, 0)]: hamming_distance(list(neighborhood.values()), list(nh.values())) for nh in neighborhoods}

                # sort distances ascending
                distances = dict(sorted(distances.items(), key=lambda k, v: v, reverse=True))

                # keep k most similar neighborhoods (k nearest neighbor neighborhoods)
                distances = dict(islice(distances, self.__k))

                # count frequency of each cell value in and pick the most common (ties broken randomly)
                cell_prediction = Counter(distances.values()).most_common(1)[0][0]

                # set the cell in the corresponding grid
                grid_predictions[gi][i, j] = cell_prediction

        return grid_predictions
