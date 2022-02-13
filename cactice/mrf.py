from typing import List

import numpy as np

from cactice.neighbors import Neighbors


class MRF:
    def __init__(self, neighbors: Neighbors = Neighbors.CARDINAL, layers: int = 1):
        """
        Initialize a Markov random field model.

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
