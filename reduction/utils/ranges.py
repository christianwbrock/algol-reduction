from .lebesgue import LebesgueSet


def closed_range(a, b):
    assert a < b
    return LebesgueSet([a, b])
