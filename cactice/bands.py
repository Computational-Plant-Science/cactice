from typing import Tuple, Dict

import numpy as np


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
