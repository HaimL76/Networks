import math
from matplotlib import pyplot as plt
import numpy as np

class Node:
    def __init__(self):
        self.neighbors: list[Node] = []

    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors.append(neighbor)

def has_double(selected_indices: np.ndarray) -> bool:
    index_set = set()

    is_double: bool = False

    index: int = 0

    while not is_double and index < len(selected_indices):
        selected_index: int = selected_indices[index]
        index += 1

        is_double = selected_index in index_set
            
        index_set.add(selected_index)
        
    return is_double

def run_ba_model(size: int, kernel_size: int):
    list_nodes: list[Node] = create_kernel(kernel_size)

    square_root_size_and_ratio: list[tuple[float, float]] = []

    for step in range(1, size):
        new_node: Node = Node()

        list_indices: list[int] = [index for index in range(len(list_nodes))]

        max_k: int = 0
        min_k: int = 0

        for i in range(len(list_nodes)):
            node: Node = list_nodes[i]

            k_i: int = len(node.neighbors)

            max_k = max(max_k, k_i)
            min_k = k_i if min_k == 0 else min(min_k, k_i)

        ratio: float = max_k / min_k if min_k > 0 else 0.0
        square_root_size: float = math.sqrt(len(list_nodes))

        square_root_size_and_ratio.append((square_root_size, ratio))

        k: int = 0

        for i in range(len(list_nodes)):
            node: Node = list_nodes[i]

            k += len(node.neighbors)

        k_avg: float = k / len(list_nodes) if len(list_nodes) > 0 else 0.0
    
        print(f"size={len(list_nodes)}, ratio={ratio}, sqrt_size={square_root_size}, k_avg={k_avg}")

        probabilities = prepare_probabilities(list_nodes, kernel_size, step)

        selected_indices = np.random.choice(list_indices, replace=False, size=kernel_size, p=probabilities)

        if has_double(selected_indices):
            raise Exception("Double selection of nodes!")

        for index in selected_indices:
            node: Node = list_nodes[index]

            node.add_neighbor(new_node)
            new_node.add_neighbor(node)

        list_nodes.append(new_node)

    sqrs: list[float] = [0.0] * len(square_root_size_and_ratio)
    ratios: list[float] = [0.0] * len(square_root_size_and_ratio)

    for i in range(len(square_root_size_and_ratio)):
        sqrs[i] = square_root_size_and_ratio[i][0]
        ratios[i] = square_root_size_and_ratio[i][1]

    plt.figure(figsize=(8, 6))
    plt.plot(sqrs, ratios, "-b")
    plt.xlabel("Square Root of N", fontsize=18)
    plt.ylabel("Max and Min Degrees Ratio", fontsize=18)
    plt.title("ba square root of size to max and min degrees ratio")
    #plt.show()
    plt.savefig("ba_model_square_root_n_ratio.png")

    dict_degrees: dict[int, int] = {}

    for node in list_nodes:
        k_i: int = len(node.neighbors)

        if k_i not in dict_degrees:
            dict_degrees[k_i] = 0

        dict_degrees[k_i] += 1

    list_degrees: list[int] = sorted(dict_degrees.keys())

    max_k: int = list_degrees[-1]
    min_k: int = list_degrees[0]

    pks: list[tuple[int, float]] = []

    degrees_count: int = sum(dict_degrees.values())

    for k in range(min_k, max_k + 1):
        count_k: int = 0
        
        if k in dict_degrees:
            count_k = dict_degrees[k]

            if count_k > 0:
                pk: float = count_k / degrees_count

                pks.append((k, pk))

    ks: list[int] = [0] * len(pks)
    pks_values: list[float] = [0.0] * len(pks)

    for i in range(len(pks)):
        tup: tuple[int, float] = pks[i]

        ks[i] = tup[0]
        pks_values[i] = tup[1]

    plt.figure(figsize=(8, 6))

    #plt.plot(node_indices, ks, "-bD")
    plt.loglog(ks, pks_values, "-b")
    plt.gca().set_xscale('log', base=np.e)
    plt.gca().set_yscale('log', base=np.e)

    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)

    plt.title("ba model P(k)")
    #plt.show()
    plt.savefig("ba_model_P_k.png")

    node_indices: list[int] = [0] * len(list_nodes)
    ks: list[int] = [0] * len(list_nodes)

    max_k: int = 0
    min_k: int = 0

    for i in range(len(list_nodes)):
        node: Node = list_nodes[i]

        k_i: int = len(node.neighbors)

        max_k = max(max_k, k_i)
        min_k = k_i if min_k == 0 else min(min_k, k_i)

        node_indices[i] = i
        ks[i] = k_i

    diff: int = max_k - min_k

    n: int = 8

    ratio: float = max_k / min_k

    exp_n: float = 1/float(n)

    alpha: float = ratio ** exp_n

    print(f"alpha={alpha}")

    bins: list[tuple[float, list[Node]]] = [[]] * (n + 1)

    for i in range(n + 1):
        threshold: float = min_k * (alpha ** i)

        print(f"threshold[{i}]={threshold}")

        bins[i] = (threshold, [])

    for node in list_nodes:
        k_i: int = len(node.neighbors)

        ratio_i: float = k_i / min_k

        log_ratio_i: float = math.log(ratio_i, alpha)

        index_bin: int = int(log_ratio_i)

        if index_bin < len(bins):
            tup: tuple[float, list[Node]] = bins[index_bin]

            list_nodes: list[Node] = tup[1]

            list_nodes.append(node)
        else:
            print("no bin found!")
            return
        
    bin_densities: dict[int, float] = {}

    for i in range(len(bins)):
        tup: tuple[float, int] = bins[i]
        next_tup: tuple[float, int] = None

        if i < len(bins) - 1:
            next_tup = bins[i + 1]

        k_i: float = tup[0]
        k_next: float = next_tup[0] if next_tup is not None else min_k * (alpha ** (n + 1))

        list_nodes: list[Node] = tup[1]

        bin_n: int = len(list_nodes)

        width = k_next - k_i

        if bin_n > 0:
            density: float = bin_n / width
            bin_densities[i] = density

    print(f"len bin_densities={len(bin_densities)}, len list degrees={len(list_degrees)}")
        
    k_bins: list[tuple[int, int, float]] = []

    for i in range(len(list_degrees)):
        k: int = list_degrees[i]

        ratio_k: float = k / min_k

        log_ratio_k: float = math.log(ratio_k, alpha)

        bin_index: int = int(log_ratio_k)

        if bin_index in bin_densities.keys():
            k_density: float = bin_densities[bin_index]

            k_bins.append((k, bin_index, k_density))
        else:
            print(f"No density for k={k} (bin_index={bin_index})")
            return

    plt.figure(figsize=(8, 6))

    xs: list[float] = [0] * len(k_bins)
    ys: list[float] = [0] * len(k_bins)

    for i in range(len(k_bins)):
        k_bin: tuple[int, int, float] = k_bins[i]

        k: int = k_bin[0]
        density: float = k_bin[2]

        xs[i] = k
        ys[i] = density

    print(f"len xs={len(xs)}, len ys={len(ys)}, len k bins={len(k_bins)}")

    plt.loglog(xs, ys, "-b")
    plt.gca().set_xscale('log', base=alpha)
    plt.gca().set_yscale('log', base=alpha)
    
    plt.xlabel("k", fontsize=18)
    plt.ylabel("density", fontsize=18)
    plt.title("ba model binned density")
    #plt.show()
    plt.savefig("ba_model_binned_density.png")

    ratio: float = max_k / min_k if min_k > 0 else 0.0
    square_root_size: float = math.sqrt(len(list_nodes))
    
    print(f"size={len(list_nodes)}, ratio={ratio}, sqrt_size={square_root_size}")

    print(f"max_k={max_k}, min_k={min_k}")

    plt.figure(figsize=(8, 6))

    #plt.plot(node_indices, ks, "-bD")
    plt.plot(node_indices, ks, "-b")

    plt.xlabel("node index", fontsize=18)
    plt.ylabel("node degree", fontsize=18)

    plt.title("ba model")
    #plt.show()
    plt.savefig("ba_model.png")


def prepare_probabilities(list_nodes: list[Node], kernel_size: int, step: int) -> np.ndarray:
    links_count: int = 0#2 * kernel_size * step

    for node in list_nodes:
        links_count += len(node.neighbors)

    probabilities: list[float] = [0.0] * len(list_nodes)

    for i in range(len(list_nodes)):
        node: Node = list_nodes[i]
        probabilities[i] = len(node.neighbors) / links_count

    return probabilities

def create_kernel(kernel_size: int) -> list[Node]:
    list_of_nodes: list[Node] = [None] * kernel_size

    for i in range(kernel_size):
        list_of_nodes[i] = Node()

    for i in range(kernel_size - 1):
        for j in range(i + 1, kernel_size):
            node_i: Node = list_of_nodes[i]
            node_j: Node = list_of_nodes[j]

            node_i.add_neighbor(node_j)
            node_j.add_neighbor(node_i)

    return list_of_nodes

run_ba_model(size=222, kernel_size=4)