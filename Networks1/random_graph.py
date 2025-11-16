import numpy as np


def construct_graph(n: int, p: float = 0.5):
    list_rows: list[list[bool]] = [[]] * n

    for i in range(n):
        list_rows[i] = [False] * n

    degrees: list[int] = [0] * n

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
            
            list_rows[i][j] = list_rows[j][i] = neighbor == 1

            degree += neighbor

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

        degrees[i] = degree

    average_degree: float = sum(degrees) / n

    list_gcc: list[set] = []

    list_gcc = collect_gcc(nodes, 0, list_gcc, None)

    print(f"n={n}, p={p:.4f}, len(list_gcc)={len(list_gcc)}")

    size_gcc: int = 0

    if isinstance(list_gcc, list) and len(list_gcc) > 0:
        for gcc in list_gcc:
            length: int = len(gcc)

            if length > size_gcc:
                size_gcc = length

        print(f"Number of nodes: {n}")
        print(f"Average degree: {average_degree:.2f}")
        print(f"GCC size: {size_gcc} ({size_gcc/n:.2%})")

def collect_gcc(nodes: dict[int, list[int]], node: int = 0,
                list_gcc: list[set] = None, gcc: set = None) -> list[set]:
    if node == 0:
        for node in nodes:
            found: bool = False
            index: int = 0

            while not found and index < len(list_gcc):
                gcc = list_gcc[index]
                index += 1

                found = node in gcc

            if not found:
                _ = collect_gcc(nodes, node, list_gcc, None)
    else:
        if not isinstance(gcc, set):
            gcc = set()
            list_gcc.append(gcc)

        if node not in gcc:
            gcc.add(node)

            print(node)

            neighbors: list[int] = nodes.get(node, [])

            for neighbor in neighbors:
                _ = collect_gcc(nodes, neighbor, list_gcc, gcc)

    return list_gcc

for i in range(5):
    construct_graph(10, p=0.0005 + i * 0.0005)
