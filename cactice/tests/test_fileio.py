from pprint import pprint

import numpy as np
import numpy.testing as npt

from cactice.fileio import read_grid_txt, read_grids_csv

tiles_path = 'testdata/grid.txt'
csv_path = 'testdata/grids.csv'


def test_read_grid_txt():
    expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                          0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                          1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1,
                          0, 0, 1, 1],
                         [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                          1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                          0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                          0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                          0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                          0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
                          0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0,
                          0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
                          0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0]])

    actual = read_grid_txt(tiles_path)
    pprint(actual)

    npt.assert_array_equal(expected, actual)


def test_read_grids_csv():
    expected = [
        # grid 34
        [[3, 0, 0, 4, 4, 3, 1, 3, 1, 3, 0, 4, 1, 0, 1, 4],
         [4, 2, 4, 0, 4, 2, 0, 2, 1, 4, 1, 2, 4, 1, 1, 3],
         [4, 0, 3, 4, 1, 1, 4, 2, 1, 4, 3, 2, 2, 0, 2, 0],
         [2, 1, 1, 0, 3, 4, 4, 1, 2, 0, 2, 3, 1, 2, 2, 3],
         [0, 4, 2, 2, 4, 0, 0, 1, 3, 1, 0, 0, 4, 2, 1, 1],
         [4, 4, 0, 3, 4, 4, 2, 2, 1, 0, 4, 4, 3, 2, 2, 2],
         [1, 4, 4, 4, 4, 4, 3, 4, 0, 4, 1, 3, 1, 1, 4, 4],
         [1, 1, 1, 4, 0, 4, 1, 1, 1, 2, 4, 1, 3, 4, 0, 4],
         [2, 2, 1, 4, 2, 1, 4, 3, 2, 1, 2, 0, 4, 2, 4, 2],
         [2, 0, 1, 4, 3, 1, 2, 1, 0, 0, 2, 2, 2, 1, 4, 1],
         [2, 2, 2, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
         [3, 1, 4, 2, 3, 2, 4, 1, 0, 1, 1, 1, 1, 4, 3, 2],
         [4, 2, 1, 2, 0, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 4, 0, 1, 4, 3, 3, 2, 3, 0, 1],
         [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 4, 3, 0, 0, 0],
         [1, 2, 4, 1, 0, 3, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0],
         [0, 0, 0, 3, 1, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0]],
        # grid 13
        [[0, 3, 1, 0, 1, 4, 2, 2, 4, 4, 4, 4, 4, 1, 1],
         [2, 4, 4, 2, 4, 3, 4, 1, 0, 0, 2, 0, 2, 0, 0]],
        # grid 12
        [[1, 1, 4, 4, 1, 4, 4, 1, 1, 4, 2, 1, 0, 1, 2, 0],
         [0, 0, 2, 2, 3, 1, 2, 0, 0, 4, 0, 3, 2, 4, 3, 4],
         [2, 1, 0, 3, 4, 1, 1, 4, 1, 1, 2, 4, 0, 4, 1, 1],
         [1, 4, 2, 1, 1, 0, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0],
         [0, 1, 2, 3, 1, 1, 0, 1, 0, 1, 1, 0, 3, 3, 2, 0]]
    ]

    actual = read_grids_csv(csv_path)
    for name, grid in actual.items():
        print(name)
        pprint(grid)

    for grid in expected:
        assert grid in [g.tolist() for g in actual.values()]
