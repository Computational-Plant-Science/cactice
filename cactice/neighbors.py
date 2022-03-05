from enum import Enum
from typing import Dict, Tuple

import numpy as np


class Neighbors(Enum):
    CARDINAL = 1  # top, bottom, left, right
    DIAGONAL = 2  # top left, top right, bottom left, bottom right
    COMPLETE = 3  # all the above


def get_neighborhood(
        grid: np.ndarray,
        i: int,
        j: int,
        neighbors: Neighbors = Neighbors.CARDINAL,
        exclude_zero: bool = False,
        absolute_coords: bool = False) -> Dict[Tuple[int, int], int]:
    """
    Gets the neighborhood around the given grid cell.

    :param grid: The grid
    :param i: The cell's row index
    :param j: The cell's column index
    :param neighbors: The cells to consider neighbors
    :param exclude_zero: Whether to exclude zero-valued neighbors
    :param absolute_coords: Use absolute coordinates rather than location relative to the central cell (the default)
    :return: A dictionary mapping cell locations to their respective values
    """

    # set the center of the neighborhood
    # neighborhood = {(0, 0): grid[i, j]}

    neighborhood = {}
    for ii in range(max(i - 1, 0), min(i + 2, grid.shape[0])):
        for jj in range(max(j - 1, 0), min(j + 2, grid.shape[1])):
            if not (i == ii and j == jj): continue  # ignore the center
            coords = (ii, jj) if absolute_coords else (ii - i, jj - j)
            if (neighbors == Neighbors.CARDINAL or neighbors == Neighbors.COMPLETE) and \
                    (i == ii and j != jj) or (i != ii and j == jj):
                neighborhood[coords] = grid[ii, jj]
            elif (neighbors == Neighbors.DIAGONAL or neighbors == Neighbors.COMPLETE) and \
                    (i != ii and j != jj):
                neighborhood[coords] = grid[ii, jj]

    # optionally exclude zeros (missing)
    if exclude_zero:
        neighborhood = {k: v for k, v in neighborhood.items() if (k == (0, 0) or (k != (0, 0) and v != 0))}

    return neighborhood


def get_neighborhoods(
        grid: np.ndarray,
        neighbors: Neighbors = Neighbors.CARDINAL,
        layers: int = 1,
        exclude_zero: bool = False):
    """
    Computes all cell neighborhoods in the given grid.

    :param grid: The grid
    :param neighbors: The cells to consider neighbors
    :param layers: The width of the neighborhood
    :param exclude_zero: Whether to exclude zero-valued neighbors
    :return: A dictionary mapping cell locations to dictionaries mapping relative locations around the central cell to neighboring values
    """
    return {(i, j): get_neighborhood(grid, i, j, neighbors, layers, exclude_zero) for i in
            range(0, grid.shape[0]) for j in range(0, grid.shape[1])}
