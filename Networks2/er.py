import math
from matplotlib import pyplot as plt
import numpy as np


def create_er(n: int, p: float):
    rows: list[list] = [[]] * n

    for i in range(n):
        rows[i] = [0] * n

    for i in range(n - 1):
        print(f"Row {i + 1}/{n}")
        size: int = n - i - 1
        
        np_arr = np.random.binomial(size=size, n=1, p=p)

        for index in range(size):
            neighbor: int = np_arr[index]

            j: int = i + 1 + index

            rows[i][j] = rows[j][i] = int(neighbor)

    ks: list[int] = [0] * n

    for i in range(n):
        neighors_counter: int = 0

        for j in range(n):
            neighbor: int = rows[i][j]

            neighors_counter += neighbor

        ks[i] = neighors_counter

        print(f"Node {i + 1}, degree={neighors_counter}")

    arr: list[int] = []

    for k in ks:
        if k >= len(arr):
            arr0: list[int] = [0] * (k + 1)

            for i in range(len(arr)):
                arr0[i] = arr[i]

            arr = arr0

        arr[k] += 1

        print(f"k={k}, P(k)={arr[k]}")

    plt.figure(figsize=(8, 6))

    xs = [k for k in range(len(arr))]
    ys = [arr[k]/n for k in range(len(arr))]

    plt.plot(xs, ys, "-bD")

    poisson_lambda: float = n * p
    poisson_arr: list[float] = [0] * len(arr)

    for k in range(len(poisson_arr)):
        poisson_k: float = math.exp(-poisson_lambda) * (poisson_lambda ** k) / math.factorial(k)
        poisson_arr[k] = poisson_k

    plt.plot(xs, poisson_arr, "-ro")

    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)

    plt.title("Degree Distribution")
    #plt.show()
    plt.savefig("degree_distribution.png")

n: int = 10000
p: float = 10 / n

create_er(n=n, p=p)