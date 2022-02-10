import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_tile(lines, target, i, j):
    rows = len(lines)
    cols = len(lines[0])

    # get this character's neighbors in the 4 cardinal directions
    up = lines[i - 1][j] if i > 0 else ' '
    down = lines[i + 1][j] if i < rows - 1 else ' '
    left = lines[i][j - 1] if j > 0 else ' '
    right = lines[i][j + 1] if j < cols - 1 else ' '

    # set the current character to empty to avoid infinite recursion
    lines[i][j] = ' '
    tile = [[i, j]]

    # function calls itself on non-empty neighbors
    if up == target: tile = tile + read_tile(lines, target, i - 1, j)
    if down == target: tile = tile + read_tile(lines, target, i + 1, j)
    if left == target: tile = tile + read_tile(lines, target, i, j - 1)
    if right == target: tile = tile + read_tile(lines, target, i, j + 1)

    # otherwise just return what we've got so far
    return tile


def read_tiles(path, target='#'):
    with open(path) as file:
        # read all lines in file
        lines = file.readlines()

        # max line length (for padding to rectangularity)
        cols = max([len(line) for line in lines])

        formatted = []
        for line in lines:
            # strip newlines
            stripped = line.replace('\n', '')

            # pad line
            padded = stripped.ljust(cols - len(stripped), ' ')

            # convert to char array
            formatted.append(list(padded))

        # find tiles
        tiles = []
        for i, line in enumerate(formatted):
            for j, char in enumerate(line):

                # if we've chanced upon a tile...
                if formatted[i][j] == target:

                    # search recursively for all target chars next to this one
                    tile = read_tile(formatted, target, i, j)

                    # save and report the find
                    tiles.append(sorted(tile))
                    logger.debug(f"Tile {len(tiles)} (size {len(tile)}): {tile}")

    logger.info(f"Found {len(tiles)} tiles")

    return sorted(tiles)


def read_csv(path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path, sep=',')

    # if there's only 1 grid and no `Grid` column, create it
    if 'Grid' not in df: df['Grid'] = 1

    # convert tabular data to list of 2D plots
    grids = dict()
    names = sorted(list(set(df['Grid'])))

    for name in names:
        # subset the data frame corresponding to the current grid
        sdf = df.loc[df['Grid'] == name]

        # find row and column counts (including missing values)
        rows = max(sdf['I']) - min(sdf['I']) + 1
        cols = max(sdf['J']) - min(sdf['J']) + 1

        # initialize grid as empty 2D array
        grid = np.zeros(shape=(rows, cols))

        # loop over cells and populate 2D array
        for i in range(0, rows):
            for j in range(0, cols):

                # check entry at location (i, j)
                matched = sdf.loc[(df['I'] == i) & (df['J'] == j)]

                # if missing, fill it in with class = 0 (unknown)
                if len(matched) == 0:
                    cls = 0
                    df = df.append({
                        'Grid': name,
                        'Class': cls,
                        'I': i,
                        'J': j
                    }, ignore_index=True)
                # otherwise use the given value
                else:
                    cls = matched.to_dict('records')[0]['Class']

                # update the grid
                grid[i, j] = cls

        # save by name and cast values from float (numpy default) to int
        grids[str(name)] = grid.astype(int)

    return grids



