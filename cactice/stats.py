from typing import Dict, Tuple
from collections import Counter, OrderedDict

import numpy as np


def class_distribution(grids: Dict[str, np.ndarray]) -> Tuple[int, int, dict, dict]:
    all_locs = [str(int(c)) for cls in [r for row in [[grid[i] for i in range(0, grid.shape[0])] for grid in grids.values()] for r in row] for c in cls]
    all_locs = [p for p in all_locs if p != '0']  # exclude missing values
    num_locs = len(all_locs)  # count total number of locations
    all_classes = list(set(all_locs))  # get unique classes
    num_classes = len(all_classes)  # count unique classes

    # count class occurrences and compute proportions
    freq = OrderedDict(sorted(dict(Counter(all_locs)).items()))
    mass = {k: round(v / sum(freq.values()), 4) for (k, v) in freq.items()}

    return num_locs, num_classes, freq, mass