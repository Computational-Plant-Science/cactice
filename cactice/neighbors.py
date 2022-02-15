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
        layers: int = 1,
        exclude_zero: bool = False) -> Dict[Tuple[int, int], int]:
    """
    Computes the neighborhood around the given grid cell.

    :param grid: The grid
    :param i: The cell's row index
    :param j: The cell's column index
    :param neighbors: The cells to consider neighbors
    :param layers: The width of the neighborhood
    :param exclude_zero: Whether to exclude zero-valued cells from neighborhoods (not counting the central cell)
    :return: A dictionary mapping relative locations from the central cell to neighboring cell values
    """

    # set the center of the neighborhood
    neighborhood = {(0, 0): grid[i, j]}

    for layer in range(1, layers + 1):
        # check if we're up against any boundaries
        bt = (i - layer < 0)                 # top
        bb = i >= (grid.shape[0] - layer)    # bottom
        bl = (j - layer < 0)                 # left
        br = j >= (grid.shape[1] - layer)    # right

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

        # TODO: compute off-cardinal/-diagonal neighbors

        # store cardinal neighbors
        if neighbors == Neighbors.CARDINAL or neighbors == Neighbors.COMPLETE:
            neighborhood[(-1, 0)] = top
            neighborhood[(1, 0)] = bottom
            neighborhood[(0, -1)] = left
            neighborhood[(0, 1)] = right

        # store diagonal neighbors
        elif neighbors == Neighbors.DIAGONAL or neighbors == Neighbors.COMPLETE:
            neighborhood[(-1, -1)] = topleft
            neighborhood[(-1, 1)] = topright
            neighborhood[(1, -1)] = bottomleft
            neighborhood[(1, 1)] = bottomright

        if neighbors == Neighbors.COMPLETE:
            # TODO: store off-cardinal/-diagonal neighbors
            pass

    # optionally exclude zero-valued neighbors
    if exclude_zero:
        neighborhood = {k: v for k, v in neighborhood.items() if (k == (0, 0) or (k != (0, 0) and v != 0))}

    # return only non-None neighbors
    return {k: v for k, v in neighborhood.items() if v is not None}

