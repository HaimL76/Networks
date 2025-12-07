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
        nodes_to_connect: list[int] = []

        while not finished and len(nodes_to_connect) < 2:
            node: int = np.random.randint(0, n)

            if node not in full_degree_nodes and node not in nodes_to_connect:
                node_neighbors: list[int] = neighbors[node]

                degree: int = 0

                index: int = 0

                while degree < d and index < n:
                    degree += node_neighbors[index]
                    index += 1

                if degree < d:
                    nodes_to_connect.append(node)
                else:
                    full_degree_nodes.add(node)

            finished = len(full_degree_nodes) >= n - d

        if len(nodes_to_connect) == 2:
            i: int = nodes_to_connect[0]
            j: int = nodes_to_connect[1]
            
            if neighbors[i][j] == 0:
                neighbors[i][j] = neighbors[j][i] = 1

                counter += 1

                print(f"d[{d}] Connected nodes {i} and {j}, total connections = {counter}")

        finished = len(full_degree_nodes) >= n - d

    gccs: list[set[int]] = collect_gcc_list(neighbors)

    gcc_size: int = 0

    if isinstance(gccs, list) and len(gccs) > 0:
        for gcc in gccs:
            if isinstance(gcc, set):
                size: int = len(gcc)
                
                if size > gcc_size:
                    gcc_size = size

    total_components = len(gccs) if isinstance(gccs, list) else 0

    tup: tuple = total_components, gcc_size

    print(f"n={n}, d={d}, total components = {total_components}, gcc size = {gcc_size}")

    return tup

def collect_gcc_list(neighbors: list[list[int]]):
    gccs: list[set[int]] = []

    for i in range(len(neighbors)):
        found: bool = False

        index: int = 0

        while not found and index < len(gccs):
            gcc: set[int] = gccs[index]

            found = i in gcc
            index += 1

        if not found:
            gcc: set[int] = set()

            gccs.append(gcc)

            collect_gcc(neighbors, i, gcc)

    return gccs

def collect_gcc(neighbors: list[list[int]], node: int, gcc: set[int], level: int = 0):
    if not isinstance(neighbors, list) or len(neighbors) < 1:
        return
    
    if not isinstance(gcc, set):
        return
    
    if level > 900:
        _ = 0

    if node not in gcc:
        gcc.add(node)

        node_neighbors: list[int] = neighbors[node]

        for j in range(len(node_neighbors)):
            neighbor: int = node_neighbors[j]

            if neighbor == 1:
                collect_gcc(neighbors, node=j, gcc=gcc, level=level + 1)
    
m = 40

p0: float = 0.00001

n: int = 2800

sys.setrecursionlimit(max(sys.getrecursionlimit(), n))

list_gcc_sizes: list[int] = [0] * m

for i in range(m):
    tup: tuple = construct_graph(n=n, d=i)
    list_gcc_sizes[i] = i, tup[1]

for i in range(m):
    print(f"d={i}, gcc size={list_gcc_sizes[i]}")

plt.figure(figsize=(8, 6))

xs = [tup[0] for tup in list_gcc_sizes]
ys = [tup[1] for tup in list_gcc_sizes]

plt.plot(xs, ys, "-bD")

plt.xlabel("d", fontsize=18)
plt.ylabel("s", fontsize=18)

plt.title("Random Graph R(n,d)")
#plt.show()
plt.savefig("random_graph_r_n_d.png")
