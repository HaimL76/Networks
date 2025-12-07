from platform import node
from random import random
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_potential_neighbors(neighbors: list[list[int]], d: int = 0, full_degree_nodes: set[int] = None):
    potential_neighbors: list[int] = []

    for i in range(len(neighbors)):
        skip: bool = isinstance(full_degree_nodes, set) and i in full_degree_nodes

        if not skip:
            row: list[int] = neighbors[i]

            j: int = 0

            counter: int = 0

            while j < len(row) and counter < d:
                counter += row[j]

                j += 1

            if counter < d:
                potential_neighbors.append(i)
            else:
                if isinstance(full_degree_nodes, set):
                    full_degree_nodes.add(i)

    return potential_neighbors


def construct_graph(n: int, d: int = 0):
    neighbors: list[list[int]] = [[]] * n

    for i in range(n):
        neighbors[i] = [0] * n

    finished: bool = False

    counter: int = 0

    full_degree_nodes: set[int] = set()

    while not finished:
        potential_neighbors: list[int] = get_potential_neighbors(neighbors, d=d, full_degree_nodes=full_degree_nodes)

        print(f"Potential neighbors: {len(potential_neighbors)}, counter: {counter}")

        finished = len(potential_neighbors) < d

        if not finished:
            len_potential_neighbors: int = len(potential_neighbors)

            i: int = np.random.randint(0, len_potential_neighbors)
            j: node = i

            while j == i:
                j: int = np.random.randint(0, len_potential_neighbors)

            node_i: int = potential_neighbors[i]
            node_j: int = potential_neighbors[j]

            neighbors[node_i][node_j] = neighbors[node_j][node_i] = 1

            counter += 1

    return neighbors    


def collect_gcc_list(nodes: dict[int, list[int]]):
    gccs: list[set[int]] = []

    for node in nodes.keys():
        found: bool = False

        index: int = 0

        while not found and index < len(gccs):
            gcc: set[int] = gccs[index]
            index += 1

            found = node in gcc

        if not found:
            gcc: set[int] = set()
            gccs.append(gcc)
            _ = collect_gcc(nodes, node, gcc)

    return gccs

def collect_gcc(nodes: dict[int, list[int]], node: int = 0,
                gcc = None, level: int = 0):
    if not isinstance(nodes, dict) or len(nodes) < 1:
        return
    
    if level > 900:
        _ = 0

    if not isinstance(gcc, set):
        gcc = set()

    if node not in gcc:
        gcc.add(node)

        neighbors: list[int] = nodes[node]

        for neighbor in neighbors:
            collect_gcc(nodes, neighbor, gcc, level + 1)

    return gcc
    
m = 10

p0: float = 0.00001

n: int = 10#int(1/p0)

sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

neighbors: list[list[int]] =construct_graph(n=n, d=3)

exit(0)

points: list[tuple[float, int]] = []

index: int = 0

finished: bool = False

while not finished:
    p: float = p0 * index
    index += 1

    for i in range(m):
        average_degree, gcc_size = construct_graph(n=n, d=3)

    finished = average_degree > 6 or p >= 1.0

plt.figure(figsize=(8, 6))

xs = [point[0] for point in points]
ys = [point[1] for point in points]

plt.plot(xs, ys, "-bD")

plt.xlabel("<k>", fontsize=18)
plt.ylabel("s", fontsize=18)

plt.title("Random Graph")
#plt.show()
plt.savefig("random_graph.png")
