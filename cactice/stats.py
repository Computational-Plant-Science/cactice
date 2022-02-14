from typing import List, Tuple, Dict
from itertools import product
from collections import Counter, OrderedDict

import numpy as np


def classes(grids: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes the probability mass function (PMF) for classes (unique values) on the given grids.

    :param grids: A list of grids
    :return: The class probability mass
    """
    cells = [str(int(c)) for cls in
             [r for row in [[grid[col_i] for col_i in range(0, grid.shape[0])] for grid in grids] for r in row] for
             c in cls]  # flatten all grid cells into single list
    cells = [c for c in cells if c != '0']  # exclude missing values

    # count occurrences and compute proportions
    freq = dict(OrderedDict(Counter(cells)))
    uniq = len(freq.keys())
    mass = {k: round(v / sum(freq.values()), uniq) for (k, v) in freq.items()}

    return mass


def undirected_bonds(grids: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Computes the probability mass function (PMF) for horizontal and vertical undirected transitions (class adjacencies) on the given grids.

    :param grids: A list of grids
    :return: A dictionary with key as random variable and value as probablity mass.
    """

    horiz = dict({
        '11': 0,
        '12': 0,
        '13': 0,
        '14': 0,
        '22': 0,
        '23': 0,
        '24': 0,
        '33': 0,
        '34': 0,
        '44': 0
    })
    vert = horiz.copy()

    for grid in grids:
        # get width and height
        w, h = grid.shape

        # count horizontal undirected transitions
        for i, j in product(range(w - 1), range(h)):
            a = grid[i, j]
            b = grid[i + 1, j]
            key = ''.join(sorted([str(int(a)), str(int(b))]))
            if '0' not in key:
                horiz[key] = horiz[key] + 1

        # count vertical undirected transitions
        for i, j in product(range(w), range(h - 1)):
            a = grid[i, j]
            b = grid[i, j + 1]
            key = ''.join(sorted([str(int(a)), str(int(b))]))
            if '0' not in key:
                vert[key] = vert[key] + 1

    # compute horizontal probability mass
    horiz_uniq = len(horiz.keys())
    horiz_sum = sum(horiz.values())
    horiz_mass = {k: round(v / horiz_sum, horiz_uniq) for (k, v) in horiz.items()} if horiz_sum > 0 else {}

    # vertical prob. mass
    vert_uniq = len(vert.keys())
    vert_sum = sum(vert.values())
    vert_mass = {k: round(v / vert_sum, vert_uniq) for (k, v) in vert.items()} if vert_sum > 0 else {}

    return horiz_mass, vert_mass
