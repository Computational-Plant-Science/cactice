from typing import List, Tuple, Dict
from itertools import product, repeat
from collections import Counter, OrderedDict
from pprint import pprint

import numpy as np


def flatten(grids: List[np.ndarray]) -> List[int]:
    """
    Flattens the given grids into a single list of cell values

    :param grids: The grids to flatten
    :return: The flattened list of grid cells
    """
    return [int(c) for cc in
             [r for row in [[grid[col_i] for col_i in range(0, grid.shape[0])] for grid in grids] for r in row] for
             c in cc]


def cell_dist(
        grids: List[np.ndarray],
        exclude_zero: bool = False) -> Dict[int, float]:
    """
    Computes the probability mass function (PMF) for classes (unique values) on the given grids.

    :param grids: A list of grids
    :param exclude_zero: Exclude zero-valued cells (interpreted to be missing values)
    :return: The class probability mass
    """

    # flatten the grids into a single list of cells
    cells = flatten(grids)

    # optionally exclude zero-valued cells
    if exclude_zero:
        cells = [cell for cell in cells if cell != 0]

    # count occurrences and compute proportions
    freq = dict(OrderedDict(Counter(cells)))
    uniq = len(freq.keys())
    mass = {k: round(v / sum(freq.values()), uniq) for (k, v) in freq.items()}

    return mass


def undirected_bond_dist(
        grids: List[np.ndarray],
        exclude_zero: bool = False) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Computes the probability mass function (PMF) for horizontal and vertical undirected transitions (class adjacencies) on the given grids.

    :param grids: A list of grids
    :param exclude_zero: Exclude zero-valued cells (interpreted to be missing values)
    :return: A dictionary with key as random variable and value as probablity mass.
    """

    # flatten the grids into a single list of cells
    cells = flatten(grids)

    # optionally exclude zero-valued cells
    if exclude_zero:
        cells = [cell for cell in cells if cell != 0]

    # enumerate undirected pairs
    classes = set(cells)
    sets = set([frozenset([p[0], p[1]]) for p in product(classes, classes)])
    pairs = sorted([(list(p) if len(p) == 2 else list(repeat(next(iter(p)), 2))) for p in sets])

    # dicts to populate
    horiz = {(ca, cb): 0 for ca, cb in pairs}
    vert = horiz.copy()

    for grid in grids:
        w, h = grid.shape

        # count horizontal bonds
        for i, j in product(range(w - 1), range(h)):
            v1 = grid[i, j]
            v2 = grid[i + 1, j]

            # optionally exclude bonds where either cell is zero-valued (missing)
            if exclude_zero and (v1 == 0 or v2 == 0):
                continue

            sk = sorted([int(v1), int(v2)])
            key = (sk[0], sk[1])
            horiz[key] = horiz[key] + 1

        # count vertical bonds
        for i, j in product(range(w), range(h - 1)):
            v1 = grid[i, j]
            v2 = grid[i, j + 1]

            # optionally exclude bonds where either cell is zero-valued (missing)
            if exclude_zero and (v1 == 0 or v2 == 0):
                continue

            sk = sorted([int(v1), int(v2)])
            key = (sk[0], sk[1])
            vert[key] = vert[key] + 1

    # compute horizontal PMF
    horiz_uniq = len(horiz.keys())
    horiz_sum = sum(horiz.values())
    horiz_mass = {k: round(v / horiz_sum, horiz_uniq) for (k, v) in horiz.items()} if horiz_sum > 0 else horiz

    # vertical PMF
    vert_uniq = len(vert.keys())
    vert_sum = sum(vert.values())
    vert_mass = {k: round(v / vert_sum, vert_uniq) for (k, v) in vert.items()} if vert_sum > 0 else vert

    return horiz_mass, vert_mass
