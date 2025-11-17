import numpy as np
import matplotlib.pyplot as plt

def construct_graph(n: int, p: float = 0.5, points: list[tuple[float, int]] = None):
    rows: list[list[bool]] = [[]] * n

    for i in range(n):
        rows[i] = [False] * n

    nodes: dict[int, list[int]] = {}

    for i in range(n - 1):
        size: int = n - 1 - i
        #print(f"Row {i+1}/{n-1}, size={size}")
        np_arr = np.random.binomial(size=size, n=1, p= p)

        arr: list[int] = np_arr.tolist()

        degree: int = 0

        for j in range(i + 1, n):
            index: int = j - (i + 1)

            neighbor: int = arr[index]
            
            rows[i][j] = rows[j][i] = neighbor == 1

            if neighbor == 1:
                node: int = i + 1
                if node not in nodes:
                    nodes[node] = []

                neighbors: list[int] = nodes[node]

                neighbor: int = j + 1
                neighbors.append(neighbor)

                if neighbor not in nodes:
                    nodes[neighbor] = []

                neighbors: list[int] = nodes[neighbor]
                neighbors.append(node)

    degrees: dict[int, int] = {}

    for i in range(n):
        row: list[bool] = rows[i]

        degree: int = 0

        for j in range(n):
            if row[j]:
                degree += 1

        node: int = i + 1

        degrees[node] = degree

    average_degree: float = 0

    for node in degrees:
        degree: int = degrees[node]
        average_degree += degree

    average_degree /= n

    gccs: list[set[int]] = collect_gcc_list(nodes)

    gcc_size: int = 0

    if isinstance(gccs, list) and len(gccs) > 0:
        for gcc in gccs:
            if isinstance(gcc, set):
                size: int = len(gcc)
                
                if size > gcc_size:
                    gcc_size = size

    total_components = len(gccs) if isinstance(gccs, list) else 0

    print(f"n={n}, p={p:.4f}, average degree = {average_degree}, gcc size = {gcc_size}, total components = {total_components}")

    tup: tuple[float, int] = average_degree, gcc_size

    if isinstance(points, list):
        points.append(tup)

    return average_degree, gcc_size

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
    
m = 200000

p0: float = 0.0001

n: int = int(0.1/p0)

points: list[tuple[float, int]] = []

index: int = 0

finished: bool = False

while not finished and index < m:
    print(f"{index}/{m}")
    p: float = p0 * index
    index += 1
    average_degree, gcc_size = construct_graph(n=n, p=p, points=points)

    finished = average_degree > 6

plt.figure(figsize=(8, 6))

xs = [point[0] for point in points]
ys = [point[1] for point in points]

plt.plot(xs, ys, "-bD")

plt.xlabel("<k>", fontsize=18)
plt.ylabel("s", fontsize=18)

plt.title("Random Graph")
#plt.show()
plt.savefig("random_graph.png")
