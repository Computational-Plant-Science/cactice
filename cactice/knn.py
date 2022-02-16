from typing import List, Dict, Tuple
from itertools import islice
from collections import Counter
import logging

import numpy as np

from cactice.neighbors import get_neighborhood, Neighbors
from cactice.distance import hamming_distance
import cactice.stats as stats


class KNN:
    def __init__(
            self,
            k: int = 10,
            neighbors: Neighbors = Neighbors.CARDINAL,
            layers: int = 1):
        """
        Create a K-nearest neighbors model.

        :param neighbors: Which adjacent cells to consider neighbors.
        :param layers: How many layers of adjacent cells to consider neighbors.
        """

        self.__logger = logging.getLogger(__name__)
        self.__k: int = k
        self.__neighbors: Neighbors = neighbors
        self.__layers: int = layers
        self.__train: List[np.ndarray] = []
        self.__cell_distribution: Dict[str, float] = {}

        # store neighborhoods as a list of grids,
        # each grid a dictionary mapping absolute cell coordinates (i, j) to neighborhoods,
        # each neighborhood a dictionary mapping relative cell coordinates (i, j) to values
        self.__neighborhoods: List[Dict[Tuple[int, int], Dict[Tuple[int, int], int]]] = []

    def fit(self, grids: List[np.ndarray]):
        """
        Fit the model to the given grids.
        """

        self.__train = grids
        self.__cell_distribution = stats.cell_dist(grids, exclude_zero=True)

        # for each grid...
        for grid in grids:
            # compute neighborhood at each cell
            neighborhoods = {(i, j): get_neighborhood(grid, i, j, self.__neighbors, self.__layers) for i in range(0, grid.shape[0]) for j in range(0, grid.shape[1])}
            self.__neighborhoods.append(neighborhoods)

    def predict(self, grids: List[np.ndarray] = None) -> List[np.ndarray]:
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
            rows = range(0, grid.shape[0])
            cols = range(0, grid.shape[1])
            grid_pred = grid_predictions[gi].copy()
            missing = [(i, j) for i in rows for j in cols if grid_pred[i, j] == 0]

            # if this grid has no missing locations, skip it
            if len(missing) == 0: continue

            # predict cells one by one
            for i, j in missing:
                # get the missing location's neighbors
                neighborhood = get_neighborhood(
                    grid=grid,
                    i=i,
                    j=j,
                    neighbors=self.__neighbors,
                    layers=self.__layers,
                    exclude_zero=True)

                # ignore central cell
                del neighborhood[(0, 0)]

                # pull out neighbor cell values
                neighbors = list(neighborhood.values())

                if len(neighbors) > 0:
                    self.__logger.debug(f"Assigning location ({i}, {j}) via KNN")

                    # compute distance from this neighborhood to every training neighborhood
                    distances = {nh[(0, 0)]: hamming_distance(list(neighborhood.values()), list(nh.values())) for nh in neighborhoods}

                    # sort distances ascending
                    distances = dict(sorted(distances.items(), key=lambda k, v: v, reverse=True))

                    # keep k most similar neighborhoods (k nearest neighbor neighborhoods)
                    distances = dict(islice(distances, self.__k))

                    # count frequency of each cell value in and pick the most common (ties broken randomly)
                    cell_pred = Counter(distances.values()).most_common(1)[0][0]
                else:
                    self.__logger.debug(
                        f"Location ({i}, {j}) has no neighbors, assigning by sampling from cell distribution")

                    # sample randomly according to cell class distribution
                    cell_pred = np.random.choice(
                        a=list(self.__cell_distribution.keys()),
                        p=list(self.__cell_distribution.values()))

                # set the cell in the corresponding grid
                grid_pred[i, j] = cell_pred

            # set the predicted grid
            grid_predictions[gi] = grid_pred

        return grid_predictions
