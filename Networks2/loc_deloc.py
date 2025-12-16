import math

import matplotlib.pyplot as plt
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


def construct_network_ring(iter: int, n: int, alpha: float, local_global: int) -> list[list[int]]:
    neighbors: list[list[int]] = [[]] * n

    for i in range(n):
        neighbors[i] = [0] * n

    for a in range(n - 1):
        for b in range(a + 1, n):
            diff: int = abs(a - b)
            mod_diff: int = n - abs(a - b)
            min_diff: int = min(diff, mod_diff)
            min_diff = diff

            p: float = 0.0
            
            if local_global == Local:
                p = math.exp(-1 * alpha * min_diff)
            elif local_global == Global:
                p = math.pow(min_diff, -alpha)
            else:
                raise ValueError("local_global must be either Local or Global")

            np_arr = np.random.binomial(size=1, n=1, p=p)

            neighbor: int = int(np_arr[0])

            neighbors[a][b] = neighbors[b][a] = neighbor

    total_degree, avg_k = calculate_average_degree(neighbors)

    lengths, avg_length = calculate_lengths(neighbors)

    print(f"[{iter}]: alpha={alpha}, Size lattice: {n}, Average degree: {avg_k}, total degree: {total_degree}, Average length: {avg_length}")

    return avg_k, avg_length

def calculate_average_degree(neighbors: list[list[int]]) -> float:
    degrees: list[int] = [0] * len(neighbors)

    n: int = len(neighbors)

    for i in range(n):
        degrees[i] = sum(neighbors[i])

    total_degree: int = sum(degrees)

    average_degree: float = total_degree / n

    return total_degree, average_degree

start: int = 1
end: int = 1000

length: int = end - start

network_length: int = 100

xs: list[float] = [0] * length
ys: list[float] = [0] * length
zs: list[float] = [0] * length

for i in range(start, end):
    alpha: float = 0.01 * i
    index: int = i - start
    avg_k, avg_length = construct_network_ring(iter=index, n=network_length, alpha=alpha, 
                                               local_global=Global)
    xs[index] = alpha
    ys[index] = avg_k
    zs[index] = avg_length

plt.figure(figsize=(8, 6))
# plot lines
plt.plot(xs, ys, label = "avg degree")
plt.plot(xs, zs, label = "avg length")
plt.legend()
plt.savefig(f"global_network_length_{network_length}_changing_alpha.png")

xs: list[float] = [0] * length
ys: list[float] = [0] * length
zs: list[float] = [0] * length

for i in range(start, end):
    alpha: float = 0.01 * i
    index: int = i - start
    avg_k, avg_length = construct_network_ring(iter=index, n=network_length, alpha=alpha, 
                                               local_global=Local)
    xs[index] = alpha
    ys[index] = avg_k
    zs[index] = avg_length

plt.figure(figsize=(8, 6))
# plot lines
plt.plot(xs, ys, label = "avg degree")
plt.plot(xs, zs, label = "avg length")
plt.legend()
plt.savefig(f"local_network_length_{network_length}_changing_alpha.png")

start = 3
end = 285

length = end - start

xs: list[float] = [0] * length
ys: list[float] = [0] * length
zs: list[float] = [0] * length

alpha: float = 0.5

for i in range(start, end):
    index: int = i - start
    avg_k, avg_length = construct_network_ring(iter=index, n=i, alpha=alpha, 
                                               local_global=Global)
    xs[index] = i
    ys[index] = avg_k
    zs[index] = avg_length

plt.figure(figsize=(8, 6))
# plot lines
plt.plot(xs, ys, label = "avg degree")
plt.plot(xs, zs, label = "avg length")
plt.legend()
plt.savefig(f"global_network_length_changing_n_alpha_{alpha}.png")

xs: list[float] = [0] * length
ys: list[float] = [0] * length
zs: list[float] = [0] * length

alpha: float = 0.5

for i in range(start, end):
    index: int = i - start
    avg_k, avg_length = construct_network_ring(iter=index, n=i, alpha=alpha, 
                                               local_global=Local)
    xs[index] = i
    ys[index] = avg_k
    zs[index] = avg_length

plt.figure(figsize=(8, 6))
# plot lines
plt.plot(xs, ys, label = "avg degree")
plt.plot(xs, zs, label = "avg length")
plt.legend()
plt.savefig(f"local_network_length_changing_n_alpha_{alpha}.png")