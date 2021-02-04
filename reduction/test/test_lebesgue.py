import random
import itertools
from math import isinf
from numpy import nextafter

from reduction.utils.lebesgue import LebesgueSet
from reduction.utils.ranges import closed_range


def _random_point():
    return random.gauss(0, 10)


def _random_set():
    points = (_random_point() for _ in range(20))
    left_infinite = random.randint(0, 1)
    return LebesgueSet(points, left_infinite)


def test_lebesgue():
    nbr_families = 10
    nbr_test_points = 100

    for i in range(nbr_families):
        all_sets = [_random_set() for _ in range(3)]
        all_set_points = set(itertools.chain(*(u.points for u in all_sets)))
        union_res = LebesgueSet.union(all_sets)
        inter_res = LebesgueSet.inter(all_sets)
        xor_res = LebesgueSet.xor(all_sets)
        for j in range(nbr_test_points):
            while True:
                x = _random_point()
                if x not in all_set_points:
                    break
            status_list = list(u.status(x) for u in all_sets)
            us_, is_, xs_ = (u.status(x)
                             for u in (union_res, inter_res, xor_res))
            u_expected = 1 if any(s == 1 for s in status_list) else -1
            i_expected = 1 if all(s == 1 for s in status_list) else -1
            x_expected = 1 if sum(s == 1 for s in status_list) & 1 else -1
            assert us_ == u_expected, "Union failed"
            assert is_ == i_expected, "Intersection failed"
            assert xs_ == x_expected, "Xor failed"


def test_ops():
    a = closed_range(1, 2)
    assert a == a
    assert a == a | a
    assert a == a & a
    assert not a ^ a
    assert a == a ^ a ^ a


def test_bool():
    r_12 = closed_range(1, 2)
    r_34 = closed_range(3, 4)
    union = r_12 | r_34
    inter = r_12 & r_34

    assert r_12
    assert r_34
    assert union
    assert not inter


def test_zoom():
    assert closed_range(1, 2).zoom(1.5, 2.0) == closed_range(0.5, 2.5)
    assert closed_range(1, 2).zoom(1.5, -2.0) == closed_range(0.5, 2.5)


def test_measure():
    assert 1 == closed_range(1, 2).measure()
    assert isinf(LebesgueSet([1, 2], left_infinite=True).measure())


def test_contains():
    a = closed_range(1, 2)

    assert nextafter(1, 0) not in a
    assert 1.0 in a
    assert 1.5 in a
    assert 2.0 in a
    assert nextafter(2, 3) not in a


def test_shifts():
    a = closed_range(1, 2)

    assert a >> 2 == closed_range(3, 4)
    assert a << 2 == closed_range(-1, 0)

    a >>= 2
    assert a == closed_range(3, 4)
