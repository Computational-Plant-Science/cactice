from typing import List

import numpy as np

from cactice.enums import Neighbors


def get_neighbors(grid, i, j, neighbors: Neighbors = Neighbors.CARDINAL, layers: int = 1):
    found = []

    # proceed outward from the nearest layer
    for layer in range(1, layers + 1):
        # check if we're up against any boundaries
        bt = (i - layer < 0)                # top
        bb = i < (grid.shape[0] - layer)    # bottom
        bl = (j - layer < 0)                # left
        br = j < (grid.shape[1] - layer)    # right

        # compute cardinal neighbors (set to None if on or beyond boundary)
        top = None if bt else grid[i - layer, j]
        bottom = None if bb else grid[i + layer, j]
        left = None if bl else grid[i, j - layer]
        right = None if br else grid[i, j + layer]

        # compute diagonal neighbors (set to None if on or beyond boundary)
        topleft = None if (bt or bl) else grid[i - layer, j - layer]
        topright = None if (bt or br) else grid[i - layer, j + layer]
        bottomleft = None if (bb or bl) else grid[i + layer, j - layer]
        bottomright = None if (bb or br) else grid[i + layer, j + layer]

        # append this layer's neighbors to the list
        if neighbors == Neighbors.CARDINAL or neighbors == Neighbors.BOUNDARY:
            found = found + [top, bottom, left, right]
        elif neighbors == Neighbors.DIAGONAL or neighbors == Neighbors.BOUNDARY:
            found = found + [topleft, topright, bottomleft, bottomright]

    # return only non-None neighbors
    return [neighbor for neighbor in found if neighbor is not None]


class KNN:
    def __init__(self, neighbors: Neighbors = Neighbors.CARDINAL, layers: int = 1):
        """
        Initialize a K-nearest neighbors model.

        :param neighbors: Which adjacent cells to consider neighbors.
        :param layers: How many layers of adjacent cells to consider neighbors.
        """
        self.__neighbors = neighbors
        self.__layers = layers

    def fit(self, grids: List[np.ndarray]):
        """
        Fit the model to the given grids (precompute neighborhoods and probability distribution).
        """
        pass

    def predict(self, grids: List[np.ndarray] = None):
        """
        Impute missing cells on the grids used to fit the model (or on the given grids, if provided).
        To generate entirely new predictions conditioned on the training set, provide a list of empty (zero) grids.

        :param grids: The grids to predict on (the training set will be used otherwise)
        :return: The grids with missing values filled in.
        """
        pass
