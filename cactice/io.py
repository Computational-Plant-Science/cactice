import logging

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


def read_file(path, target='#'):
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
