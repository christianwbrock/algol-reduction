from .lebesgue import LebesgueSet
import numpy as np


def closed_range(a, b):
    assert a < b
    return LebesgueSet([a, b])


def union_of_ranges(lst):
    """
    convert a list of range boundaries to a LebesgueSet
    """

    if isinstance(lst, LebesgueSet):
        return lst

    assert not lst or np.ndim(lst) == 2
    assert not lst or np.shape(lst)[1] >= 2

    result = None

    for lower, upper in lst:
        item = closed_range(lower, upper)
        result = result | item if result else item

    return result
