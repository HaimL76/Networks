import math

import numpy as np

Local: int = 0
Global: int = 1

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

    print(f"Size lattice: {n}, Average degree: {avg_k}")

def calculate_average_degree(neighbors: list[list[int]]) -> float:
    degrees: list[int] = [0] * len(neighbors)

    n: int = len(neighbors)

    for i in range(n):
        degrees[i] = sum(neighbors[i])

    total_degree: int = sum(degrees)

    average_degree: float = total_degree / n

    return average_degree

for i in range(1000):
    construct_network_ring(i + 1, 0.5, Local)