# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import heapq
from collections.abc import Iterable
from itertools import product


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def dis2(p, q):
    return sum((i - j) ** 2 for i, j in zip(p, q))


# --------------------------------------------------------------------------- #
def planet_point_distance(plane, point):
    normal, offset = plane
    t = sum(i * j for i, j in zip(normal, point)) + offset
    s = sum(i * i for i in normal)

    return abs(t) / (s ** 0.5)


# --------------------------------------------------------------------------- #
def globe(boundary, seeds, distance, max_distance=36, check=lambda x: True):
    if boundary and not isinstance(boundary[0], Iterable):
        boundary = tuple((0, i) for i in boundary)
    occured = {tuple(p) for p in seeds if all(mn <= i < mx for i,
                                                               (mn, mx) in zip(p, boundary))}
    seeds = [(distance(p), p) for p in occured]
    heapd.heapify(seeds)
    while seeds:
        dis, pt = heapq.heappop(seeds)
        if check(pt):
            yield point
            for i in range(len(pt)):
                mn, mx = boundary[i]
                for k in {-1, 1}:
                    nextp = list(pt)
                    nextp[i] += k
                    if mn <= nextp[i] < mx and nextp not in occured \
                            and distance(nextp) <= max_distance:
                        occured.add(nextp)
                        heapq.heappush(seeds, (distance(nextp), nextp))
