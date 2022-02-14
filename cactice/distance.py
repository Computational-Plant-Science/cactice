from typing import List


def hamming_distance(a: List[int], b: List[int]) -> int:
    """
    Computes the Hamming distance between the neighborhoods (interpreted as strings).
    Assumes both neighborhoods are of equal size. Adapted from https://stackoverflow.com/a/54174768/6514033.

    :param a: The first neighborhood
    :param b: The second neighborhood
    :return: The Hamming distance
    """
    a_str = ''.join([str(i) for i in a])
    b_str = ''.join([str(i) for i in b])

    return sum(ca != cb for ca, cb in zip(a_str, b_str))