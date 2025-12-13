import math

import numpy as np

Local: int = 0
Global: int = 1

def calculate_lengths(neighbors: list[list[int]],
                      should_print: bool = False):
    np_neighbors_matrix: np.ndarray = np.array(neighbors)

    dim: int = len(neighbors)

    np_lengths_matrix: np.ndarray = np.identity(dim, dtype=int)

    lengths: list[list[int]] = [[]] * dim

    for i in range(dim):
        lengths[i] = [0] * dim

    finished: bool = False

    length: int = 0

    while not finished:
        length += 1

        np_lengths_matrix = np.matmul(np_lengths_matrix, np_neighbors_matrix)

        counter: int = 0

        for i in range(dim - 1):
            for j in range(i + 1, dim):
                if lengths[i][j] == 0 and np_lengths_matrix[i][j] > 0:
                    lengths[i][j] = lengths[j][i] = length
                    counter += 1

        finished = counter < 1

    counter = 0 # dummy

    length = 0

    for i in range(dim - 1):
        for j in range(i + 1, dim):
            length += lengths[i][j]
            counter += 1

    average_length: float = float(length) / counter

    return lengths, average_length


def construct_network_ring(n: int, alpha: float, local_global: int) -> list[list[int]]:
    neighbors: list[list[int]] = [[]] * n

    for i in range(n):
        neighbors[i] = [0] * n

    for a in range(n - 1):
        for b in range(a + 1, n):
            p: float = math.exp(-1 * alpha * abs(a - b)) if local_global == Local else math.pow(abs(a - b), -alpha)

            #print(f"Nodes: ({a}, {b}), Probability: {p}")

            np_arr = np.random.binomial(size=1, n=1, p=p)

            neighbor: int = int(np_arr[0])

            neighbors[a][b] = neighbors[b][a] = neighbor

    avg_k: float = calculate_average_degree(neighbors)

    lengths, avg_length = calculate_lengths(neighbors)

    print(f"Size lattice: {n}, Average degree: {avg_k}, Average length: {avg_length}")

def calculate_average_degree(neighbors: list[list[int]]) -> float:
    degrees: list[int] = [0] * len(neighbors)

    n: int = len(neighbors)

    for i in range(n):
        degrees[i] = sum(neighbors[i])

    total_degree: int = sum(degrees)

    average_degree: float = total_degree / n

    return average_degree

for i in range(3, 1000):
    construct_network_ring(i, 0.5, Local)