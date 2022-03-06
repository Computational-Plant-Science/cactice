import numpy as np
from pprint import pprint

from matplotlib.patches import Rectangle

from cactice.bands import get_band
from cactice.fileio import read_grids_csv
from cactice.plot import plot_grid

grids = read_grids_csv('testdata/grids.csv')


def test_get_band():
    grid = grids['4']
    cell = (5, 8)

    # TODO: if we allow start=0, reinstate this
    # trivial case
    # start = 0
    # band = get_band(grid, i=cell[0], j=cell[1], start=start)
    # assert len(band.keys()) == 1
    # assert band[(start, start)] == grid[(cell[0], cell[1])]

    # band = get_band(grid, i=cell[0], j=cell[1], start=start, absolute_coords=True)
    # assert len(band.keys()) == 1
    # assert band[(cell[0], cell[1])] == grid[(cell[0], cell[1])]

    # immediate cardinal+diagonal neighborhood
    start = 1

    band = get_band(grid, i=cell[0], j=cell[1], start=start)
    assert len(band.keys()) == 8
    assert band[(start, start)] == grid[(cell[0] + start, cell[1] + start)]
    assert band[(-1 * start, -1 * start)] == grid[(cell[0] - start, cell[1] - start)]

    band = get_band(grid, i=cell[0], j=cell[1], start=start, absolute_coords=True)
    side = 2 * start + 1
    # debugging
    # plot_grid(grid, patch=Rectangle((cell[0] - start, cell[1] - start), side, side, fill=False, edgecolor='blue', lw=3))
    assert len(band.keys()) == 8
    assert band[(cell[0] + start, cell[1] + start)] == grid[(cell[0] + start, cell[1] + start)]
    assert band[(cell[0] - start, cell[1] - start)] == grid[(cell[0] - start, cell[1] - start)]

    # another layer out
    start = 2

    band = get_band(grid, i=cell[0], j=cell[1], start=start)
    assert len(band.keys()) == 16
    assert band[(start, start)] == grid[(cell[0] + start, cell[1] + start)]
    assert band[(-1 * start, -1 * start)] == grid[(cell[0] - start, cell[1] - start)]

    band = get_band(grid, i=cell[0], j=cell[1], start=start, absolute_coords=True)
    assert len(band.keys()) == 16
    assert band[(cell[0] + start, cell[1] + start)] == grid[(cell[0] + start, cell[1] + start)]
    assert band[(cell[0] - start, cell[1] - start)] == grid[(cell[0] - start, cell[1] - start)]



