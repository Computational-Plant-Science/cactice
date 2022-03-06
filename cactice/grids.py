import logging
from enum import Enum
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Neighbors(Enum):
    CARDINAL = 1  # top, bottom, left, right
    DIAGONAL = 2  # top left, top right, bottom left, bottom right
    COMPLETE = 3  # all the above


def get_neighborhood(
        grid: np.ndarray,
        i: int,
        j: int,
        neighbors: Neighbors = Neighbors.CARDINAL,
        include_center: bool = False,
        exclude_zero: bool = False,
        absolute_coords: bool = False) -> Dict[Tuple[int, int], int]:
    """
    Gets the neighborhood around the given grid cell.

    :param grid: The grid
    :param i: The cell's row index
    :param j: The cell's column index
    :param neighbors: The cells to consider neighbors
    :param include_center: Whether to include the central cell in the neighborhood
    :param exclude_zero: Whether to exclude zero-valued neighbors
    :param absolute_coords: Use absolute coordinates rather than location relative to the central cell (the default)
    :return: A dictionary mapping cell locations to their respective values
    """

    # optionally include the central cell in the neighborhood we'll return
    neighborhood = {(0, 0): grid[i, j]} if include_center else {}

    irange = (max(i - 1, 0), min(i + 1, grid.shape[0]))
    jrange = (max(j - 1, 0), min(j + 1, grid.shape[1]))
    for ii in range(irange[0], irange[1] + 1):
        for jj in range(jrange[0], jrange[1] + 1):
            if i == ii and j == jj: continue                                # ignore the center
            if ii >= grid.shape[0] or jj >= grid.shape[1]: continue         # make sure we're still within the grid
            coords = (ii, jj) if absolute_coords else (ii - i, jj - j)      # use relative or absolute coordinates
            if (neighbors == Neighbors.DIAGONAL or neighbors == Neighbors.COMPLETE) \
                    and (i != ii and j != jj):                              # diagonals: both coords are different
                logger.info(f"Adding diagonal neighbor ({ii}, {jj}), ({i}, {j})")
                neighborhood[coords] = grid[ii, jj]
            elif (neighbors == Neighbors.CARDINAL or neighbors == Neighbors.COMPLETE) \
                    and ((i == ii and j != jj) or (i != ii and j == jj)):   # cardinals: 1 coord equal, 1 different
                logger.info(f"Adding cardinal neighbor ({ii}, {jj}), ({i}, {j})")
                neighborhood[coords] = grid[ii, jj]

    # optionally exclude zeros (interpreted as missing values)
    if exclude_zero:
        neighborhood = {k: v for k, v in neighborhood.items() if (k == (0, 0) or (k != (0, 0) and v != 0))}

    return neighborhood


def get_neighborhoods(
        grid: np.ndarray,
        neighbors: Neighbors = Neighbors.CARDINAL,
        include_center: bool = False,
        exclude_zero: bool = False):
    """
    Computes all cell neighborhoods in the given grid.

    :param grid: The grid
    :param neighbors: The cells to consider neighbors
    :param include_center: Whether to include the central cell in the neighborhood
    :param exclude_zero: Whether to exclude zero-valued neighbors
    :return: A dictionary mapping cell locations to dictionaries mapping relative locations around the central cell to neighboring values
    """
    return {(i, j): get_neighborhood(grid, i, j, neighbors, include_center, exclude_zero) for i in
            range(0, grid.shape[0]) for j in range(0, grid.shape[1])}


def get_band(
        grid: np.ndarray,
        i: int,
        j: int,
        start: int,
        exclude_zero: bool = False,
        absolute_coords: bool = False) -> Dict[Tuple[int, int], int]:
    """

    :param grid: The grid
    :param i: The central cell's row index
    :param j: The central cell's column index
    :param start: The distance from the central cell to start the band
    :param exclude_zero: Whether to exclude zero-valued cells
    :param absolute_coords: Use absolute coordinates rather than location relative to the central cell (the default)
    :return: A dictionary mapping cell locations to their respective values
    """

    if start < 1 or start > min(grid.shape):
        raise ValueError(f"Band starting distance must be greater than 0 and less than min(grid length, grid width)")

    band = {}
    ir = (max(i - start, 0), min(i + start, grid.shape[0]))
    jr = (max(j - start, 0), min(j + start, grid.shape[1]))
    for ii in range(ir[0], ir[1] + 1):
        for jj in range(jr[0], jr[1] + 1):
            # skip interior cells
            if (ii != ir[0] and ii != ir[1]) and (jj != jr[0] and jj != jr[1]):
                print(f"Skipping {ii}, {jj}")
                continue

            # map the cell's value to relative or absolute coordinates
            coords = (ii, jj) if absolute_coords else (ii - i, jj - j)
            band[coords] = grid[ii, jj]

    # optionally exclude zeros (missing values)
    if exclude_zero:
        band = {k: v for k, v in band.items() if (k == (0, 0) or (k != (0, 0) and v != 0))}

    return band
