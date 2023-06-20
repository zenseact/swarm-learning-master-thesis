import itertools
from typing import List


def fully_connected_centralised(clients: List[str]) -> None:
    # returns a list of all possible edges in a fully connected directed graph
    return list(itertools.permutations(clients, 2))


def edgeless_graph(clients: List[str]) -> None:
    # returns an empty list representing an edgeless graph
    return []
