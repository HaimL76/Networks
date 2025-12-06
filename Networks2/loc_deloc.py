import math

import numpy as np


def construct_network_ring(n: int):
    neighbors: list[list[int]] = [[]] * n

    for i in range(n):
        neighbors[i] = [0] * n

    for a in range(n - 1):
        for b in range(a + 1, n):
            p: float = math.exp(-1 * a * abs(a - b))

            np_arr = np.random.binomial(size=1, n=1, p=p)

            neighbor: int = int(np_arr[0])

            neighbors[a][b] = neighbors[b][a] = neighbor

    for 