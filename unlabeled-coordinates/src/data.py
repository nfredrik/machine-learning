"""Utilities to work with our sample domain data."""

from typing import List
from itertools import groupby
import operator


Vector = List[float]
Cluster = List[Vector]


def vectors_to_arrays(data: List[dict]) -> List[Cluster]:
    """Translates json encoded vector data to arrays.

    Also arrays are grouped by class.

    E.g.
        [
            {'x': 1, 'y': 1, 'class': 1},
            {'x': -1, 'y': -1, 'class': 2},
            {'x': 2, 'y': 2, 'class': 1},
        ]

        To:
        [
            [[1, 1], [2, 2]],
            [[-1, -1]],
        ]
    """
    return [[[v['x'], v['y']] for v in data]
            for _, data in vectors_by_class(data)]


def vectors_by_class(data: List[dict]) -> list:
    sort_key = operator.itemgetter('class')
    return groupby(sorted(data, key=sort_key), key=sort_key)
