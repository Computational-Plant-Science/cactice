import sys
import random
from typing import List

import numpy as np
import pandas as pd

from cactice.enums import Neighbors


# TODO: more distance function options?

def hamming_distance(a: List[int], b: List[int]) -> int:
    """
    Computes the Hamming distance between the neighborhoods (interpreted as strings).
    Assumes both neighborhoods are of equal size. Adapted from https://stackoverflow.com/a/54174768/6514033.

    :param a: The first neighborhood
    :param b: The second neighborhood
    :return: The Hamming distance
    """
    a_str = ''.join([str(i) for i in a])
    b_str = ''.join([str(i) for i in b])

    return sum(ca != cb for ca, cb in zip(a_str, b_str))


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
        if neighbors == Neighbors.CARDINAL or neighbors == Neighbors.COMPLETE:
            found = found + [top, bottom, left, right]
        elif neighbors == Neighbors.DIAGONAL or neighbors == Neighbors.COMPLETE:
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
        self.__neighborhoods: List[List[int]] = []

    def fit(self, grids: List[np.ndarray]):
        """
        Fit the model to the given grids (precompute neighborhoods).
        """

        # set the training set
        self.__train = grids

        # compute neighbors for each cell in each grid
        for grid in grids:
            self.__neighborhoods = self.__neighborhoods + [[grid[i, j]] + get_neighbors(grid, i, j, self.__neighbors, self.__layers) for i in range(0, grid.shape[0]) for j in range(0, grid.shape[1])]

        # TODO: is there any reason to use a data frame?
        # create data frame headers
        # columns = ['Cell']
        # if self.__neighbors == Neighbors.CARDINAL or self.__neighbors == Neighbors.COMPLETE:
        #     columns = columns + np.ravel([[f"Top{l}", f"Bottom{l}", f"Left{l}", f"Right{l}"] for l in range(1, self.__layers + 1)])
        # if self.__neighbors == Neighbors.DIAGONAL or self.__neighbors == Neighbors.COMPLETE:
        #     columns = columns + np.ravel([[f"Topleft{l}", f"Topright{l}", f"Bottomleft{l}", f"Bottomright{l}"] for l in range(1, self.__layers + 1)])

        # self.__neighborhoods = pd.DataFrame(data=neighborhoods, columns=columns, dtype=int)

    def predict(self, grids: List[np.ndarray] = None):
        """
        Impute missing cells on the grids used to fit the model (or on the given grids, if provided).
        To generate entirely new predictions conditioned on the training set, provide a list of empty (zero) grids.

        :param grids: The grids to predict on (the training set will be used otherwise)
        :return: The grids with missing values filled in.
        """

        # initialize grids to predict on
        prediction_grids = [np.copy(grid) for grid in (grids if grids is not None else self.__train)]

        for grid_index, grid in enumerate(prediction_grids):
            # find cells to predict (missing locations)
            missing = [(i, j) for i in range(0, grid.shape[0]) for j in range(0, grid.shape[1]) if grid[i, j] == 0]

            # predict cells one by one
            for i, j in missing:
                prediction = 0
                min_dist = sys.float_info.max

                # get the missing location's neighbors
                neighbors = get_neighbors(grid, i, j, self.__neighbors, self.__layers)

                # compute distance from this neighborhood to every training neighborhood
                for neighborhood in self.__neighborhoods:
                    distance = hamming_distance(neighbors, neighborhood[1:])

                    # if we have a new minimum distance, update the prediction
                    if distance < min_dist:
                        min_dist = distance
                        prediction = neighborhood[0]

                # set the cell in the corresponding grid
                prediction_grids[grid_index][i, j] = prediction

        return prediction_grids
