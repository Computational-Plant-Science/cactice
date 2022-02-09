from pprint import pprint

from cactice.io import read_tiles, read_tile

path = 'testdata/sample1.txt'


def test_read_tile():
    pass


def test_read_file():
    expected = sorted([
        sorted([[0, 28], [1, 28], [2, 28], [3, 28], [2, 29]]),
        sorted([[1, 2], [1, 3], [1, 4], [2, 3], [3, 3]]),
        sorted([[1, 13], [2, 13], [3, 13], [3, 14], [3, 15]]),
        sorted([[1, 18], [2, 18], [2, 19], [3, 19], [3, 20]]),
        sorted([[1, 24], [2, 23], [2, 24], [2, 25], [3, 24]]),
        sorted([[2, 7], [2, 9], [3, 7], [3, 8], [3, 9]]),
        sorted([[5, 7], [6, 7], [7, 7], [8, 7], [9, 7]]),
        sorted([[5, 10], [6, 10], [7, 10], [8, 10], [8, 11]]),
        sorted([[5, 19], [6, 19], [7, 18], [7, 19], [8, 18]]),
        sorted([[6, 3], [6, 4], [7, 2], [7, 3], [8, 3]]),
        sorted([[6, 14], [6, 15], [6, 15], [7, 14], [7, 15], [8, 14]]),
        sorted([[6, 22], [6, 23], [7, 23], [8, 23], [8, 24]])
    ])

    actual = read_tiles(path)
    # pprint(actual)

    for tile in expected: assert tile in actual
    for tile in actual: assert tile in expected
