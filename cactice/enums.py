from enum import Enum


class Neighbors(Enum):
    CARDINAL = 1  # top, bottom, left, right
    DIAGONAL = 2  # top left, top right, bottom left, bottom right
    BOUNDARY = 3  # all the above