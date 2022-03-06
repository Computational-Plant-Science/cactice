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
            # ignore the center
            if i == ii and j == jj:
                continue

            # make sure we're still within the grid
            if ii >= grid.shape[0] or jj >= grid.shape[1]:
                continue

            # use relative or absolute coordinates
            coords = (ii, jj) if absolute_coords else (ii - i, jj - j)

            # diagonals: both coords are different
            if (neighbors == Neighbors.DIAGONAL or neighbors == Neighbors.COMPLETE) \
                    and (i != ii and j != jj):
                logger.info(f"Adding cell ({i}, {j})'s diagonal neighbor ({ii}, {jj})")
                neighborhood[coords] = grid[ii, jj]

            # cardinals: 1 coord equal, 1 different
            elif (neighbors == Neighbors.CARDINAL or neighbors == Neighbors.COMPLETE) \
                    and ((i == ii and j != jj) or (i != ii and j == jj)):
                logger.info(f"Adding cell ({i}, {j})'s cardinal neighbor ({ii}, {jj}), ({i}, {j})")
                neighborhood[coords] = grid[ii, jj]

    # optionally exclude zeros (interpreted as missing values)
    if exclude_zero:
        neighborhood = {k: v for k, v in neighborhood.items() if (k == (0, 0) or (k != (0, 0) and v != 0))}

    return neighborhood


def get_neighborhoods(
        grid: np.ndarray,
        neighbors: Neighbors = Neighbors.CARDINAL,
        include_center: bool = False,
        exclude_zero: bool = False,
        absolute_coords: bool = False):
    """
    Computes all cell neighborhoods in the given grid.

    :param grid: The grid
    :param neighbors: The cells to consider neighbors
    :param include_center: Whether to include the central cell in the neighborhood
    :param exclude_zero: Whether to exclude zero-valued neighbors
    :param absolute_coords: Use absolute coordinates rather than location relative to the central cell (the default)
    :return: A dictionary mapping cell locations to dictionaries mapping relative locations around the central cell to neighboring values
    """

    return {(i, j): get_neighborhood(grid=grid,
                                     i=i,
                                     j=j,
                                     neighbors=neighbors,
                                     include_center=include_center,
                                     exclude_zero=exclude_zero,
                                     absolute_coords=absolute_coords)
            for i in range(0, grid.shape[0])
            for j in range(0, grid.shape[1])}


def get_band(
        grid: np.ndarray,
        i: int,
        j: int,
        distance: int = 1,
        include_center: bool = False,
        exclude_zero: bool = False,
        absolute_coords: bool = False) -> Dict[Tuple[int, int], int]:
    """
    Compute the (square) band at the given distance around the given cell location.

    :param grid: The grid
    :param i: The central cell's row index
    :param j: The central cell's column index
    :param distance: The distance from the central cell to the band
    :param include_center: Whether to include the central cell in the neighborhood
    :param exclude_zero: Whether to exclude zero-valued cells
    :param absolute_coords: Use absolute coordinates rather than location relative to the central cell (the default)
    :return: A dictionary mapping cell locations to their respective values
    """

    if distance < 1 or distance > min(grid.shape):
        raise ValueError(f"Band distance must be greater than 0 and less than min(grid length, grid width)")

    band = {(0, 0): grid[i, j]} if include_center else {}
    ir = (max(i - distance, 0), min(i + distance, grid.shape[0]))
    jr = (max(j - distance, 0), min(j + distance, grid.shape[1]))
    for ii in range(ir[0], ir[1] + 1):
        for jj in range(jr[0], jr[1] + 1):
            # skip interior cells
            if (ii != ir[0] and ii != ir[1]) and (jj != jr[0] and jj != jr[1]):
                continue

            # make sure we're still within the grid
            if ii >= grid.shape[0] or jj >= grid.shape[1]:
                continue

            # map the cell's value to relative or absolute coordinates
            logger.info(f"Adding cell ({i}, {j})'s band cell ({ii}, {jj})")
            coords = (ii, jj) if absolute_coords else (ii - i, jj - j)
            band[coords] = grid[ii, jj]

    # optionally exclude zeros (missing values)
    if exclude_zero:
        band = {k: v for k, v in band.items() if (k == (0, 0) or (k != (0, 0) and v != 0))}

    return band


def get_bands(
        grid: np.ndarray,
        distance: int = 1,
        include_center: bool = False,
        exclude_zero: bool = False,
        absolute_coords: bool = False) -> Dict[Tuple[int, int], int]:
    """
    Computes all bands at the given distance in the given grid.

    :param grid: The grid
    :param distance: The distance from the central cell to start the band
    :param include_center: Whether to include the central cell in the neighborhood
    :param exclude_zero: Whether to exclude zero-valued cells
    :param absolute_coords: Use absolute coordinates rather than location relative to the central cell (the default)
    :return: A dictionary mapping cell locations to dictionaries mapping band cell locations to their respective values
    """

    if distance < 1 or distance > min(grid.shape):
        raise ValueError(f"Band distance must be greater than 0 and less than min(grid length, grid width)")

    return {(i, j): get_band(grid=grid,
                             distance=distance,
                             i=i,
                             j=j,
                             include_center=include_center,
                             exclude_zero=exclude_zero,
                             absolute_coords=absolute_coords)
            for i in range(0, grid.shape[0])
            for j in range(0, grid.shape[1])}
