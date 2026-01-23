import math
import os
from matplotlib import pyplot as plt
import numpy as np

class Node:
    def __init__(self, step: int, fitness: float = 0.0):
        self.appearance_step: int = step
        self.fitness: float = fitness
        self.neighbors: list[Node] = []

    def add_neighbor(self, neighbor: 'Node'):
        self.neighbors.append(neighbor)

    def __str__(self):
        num_neighbors: int = 0
        
        if isinstance(self.neighbors, list):
            num_neighbors = len(self.neighbors)
        
        return str(num_neighbors)

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

def run_ba_model(num_steps: int, kernel_size: int, fitness: tuple = None):
    os.makedirs("ba_figs", exist_ok=True)

    fitness_param: float = 0.0
    fitness_method: str = None

    if isinstance(fitness, tuple) and len(fitness) == 2:
        fitness_method = fitness[0]
        fitness_param = fitness[1]

    with_fitness: str = f"{fitness_method}_{fitness_param}" if fitness_method and fitness_param > 0.0 else ""

    kernel: list[Node] = create_kernel(kernel_size)

    node_index: int = len(kernel)

    nodes_count: int = node_index + num_steps

    list_nodes: list[Node]  = [None] * nodes_count

    list_nodes[:node_index] = kernel
    
    num_steps_to_track: int = 2
    
    ki_by_time: list[tuple[int, list[int]]] = [None] * num_steps_to_track

    num_list_nodes: int = len(list_nodes)

    diff: int = int(num_list_nodes / num_steps_to_track)

    for i in range(num_steps_to_track):
        ti: int = i * diff
        if ti == 0:
            ti = 1

        ki_by_time[i] = (ti, [])

    k: int = kernel_size - 1

    k_max: int = k
    k_min: int = k

    k_ratio_n: list[tuple[float, float]] = [None] * nodes_count
    k_average: list[float] = [0.0] * nodes_count

    k_average_kernel: float = k

    k_ratio_n[:node_index] = [(node_index, 1) for _ in kernel]
    k_average[:node_index] = [k_average_kernel for _ in kernel]

    for step in range(num_steps):
        curr_node_index: int = node_index
        node_index += 1

        step_id: int = step + 1

        fitness_value: float = 0.0

        new_node: Node = Node(step=step_id, fitness=fitness_value)

        list_nodes[curr_node_index] = new_node

        probabilities = prepare_probabilities(list_nodes, node_index=node_index, step=step)
        
        list_indices: list[int] = [index for index in range(len(probabilities))]

        selected_indices = np.random.choice(list_indices, replace=False, size=kernel_size, p=probabilities)

        if has_double(selected_indices):
            raise Exception("double selection of nodes")

        for index in selected_indices:
            node: Node = list_nodes[index]

            node.add_neighbor(new_node)
            new_node.add_neighbor(node)

            k += 2

            node_k: int = len(node.neighbors)
            new_node_k: int = len(new_node.neighbors)

            k_max: int = max(k_max, node_k, new_node_k)
            k_min: int = min(k_min, node_k, new_node_k)

            k_ratio: float = k_max / k_min

            k_ratio_n[curr_node_index] = (curr_node_index ** 0.5, k_ratio)
            k_average[curr_node_index] = k / curr_node_index

        print(f"step={step}, new_node_degree={len(new_node.neighbors)}, max_k={k_max}, min_k={k_min}")

        for i in range(len(ki_by_time)):
            tup: tuple[int, list[int]] = ki_by_time[i]

            ti: int = tup[0]

            node: Node = list_nodes[ti]
            
            ki_for_node: list[int] = tup[1]

            if not ki_for_node:
                ki_for_node = [0] * num_steps
                tup = (ti, ki_for_node)
                ki_by_time[i] = tup

            ki_for_node = tup[1]

            if node is None:
                ki_for_node[step] = 0
            else:
                ki_for_node[step] = len(node.neighbors)

    save_square_root_n_ratio_plot(ki_by_time=ki_by_time,
                                  kernel_size=kernel_size,
                                  with_fitness=with_fitness)

    save_k_average_n_plot(kernel_size=kernel_size, k_average=k_average, 
                          with_fitness=with_fitness)
    save_k_ratio_n_plot(k_ratio_n=k_ratio_n, with_fitness=with_fitness)

    dict_k: dict[int, int] = {}

    for node in list_nodes:
        k_i: int = len(node.neighbors)

        if k_i not in dict_k:
            dict_k[k_i] = 0

        dict_k[k_i] += 1

    save_p_k_plot(dict_k=dict_k, nodes_count=nodes_count, 
                  with_fitness=with_fitness)
    
def save_square_root_n_ratio_plot(ki_by_time: list[tuple[int, list[int]]],
                                  kernel_size: int,
                                  with_fitness: str):
    
    num_steps: int = len(ki_by_time[0][1])

    xs: list[int] = [0.0] * num_steps
    list_ys: list[list[tuple[float, float]]] = [[]] * num_steps

    plt.figure(figsize=(8, 6))
    plt.xlabel("t", fontsize=18)
    plt.ylabel("k_i and sqrt(t)", fontsize=18)
    plt.title("ba model k_i and sqrt(t) by t")
    plt.legend(["k_i", "sqrt(t)"])

    for i in range(len(ki_by_time)):
        tup: tuple[int, list[int]] = ki_by_time[i]

        ti: int = tup[0]
        list_ki: list[int] = tup[1]

        ys_ki: list[float] = [0.0] * len(list_ki)
        ys_calc_ki: list[float] = [0.0] * len(list_ki)

        for t in range(len(list_ki)):
            xs[t] = t

            calc_ki: float = kernel_size * ((t/ti) ** 0.5)
            
            ki: int = list_ki[t]

            ys_ki[t] = ki
            ys_calc_ki[t] = calc_ki

        plt.loglog(xs, ys_ki, "-b")
        plt.loglog(xs, ys_calc_ki, "-r")

    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_k_i_sqrt_t_loglog{('_with_fitness_' + with_fitness) if with_fitness else ''}.png")

def save_p_k_plot(dict_k: dict[int, int], nodes_count: int, 
                  with_fitness: str):
    ks: list[int] = sorted(dict_k.keys())

    xs: list[int] = [0] * len(ks)
    ys: list[float] = [0.0] * len(ks)

    for i in range(len(ks)):
        k: int = ks[i]
        count_k: int = dict_k[k]

        xs[i] = k
        ys[i] = count_k / nodes_count

    plt.figure(figsize=(8, 6))
    plt.loglog(xs, ys, "-b")
    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)
    plt.title("ba model P(k)")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_p_k_loglog{('_with_fitness_' + with_fitness) if with_fitness else ''}.png")

def save_k_average_n_plot(kernel_size: int, k_average: list[float], 
                          with_fitness: str):
    xs: list[int] = [0] * len(k_average)
    ys: list[float] = [0.0] * len(k_average)
    ys1: list[int] = [0] * len(k_average)

    for i in range(len(k_average)):
        xs[i] = i
        ys[i] = k_average[i]
        ys1[i] = 2 * kernel_size

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, "-b")
    plt.plot(xs, ys1, "-r")
    plt.xlabel("N", fontsize=18)
    plt.ylabel("Average Degree", fontsize=18)
    plt.title("ba model average degree by N")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_k_average_n{('_with_fitness_' + with_fitness) if with_fitness else ''}.png")

def save_k_ratio_n_plot(k_ratio_n: list[tuple[int, float]], with_fitness: str):
    xs: list[int] = [0] * len(k_ratio_n)
    ys: list[float] = [0.0] * len(k_ratio_n)

    for i in range(len(k_ratio_n)):
        tup: tuple[int, float] = k_ratio_n[i]

        xs[i] = tup[0]
        ys[i] = tup[1]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, "-b")
    plt.xlabel("N", fontsize=18)
    plt.ylabel("Max and Min Degrees Ratio", fontsize=18)
    plt.title("ba model max and min degrees ratio by N")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_k_ratio_n{('_with_fitness_' + with_fitness) if with_fitness else ''}.png")

def kuku():
    file_name_fitness_part: str = f"_with_fitness_{with_fitness}" if with_fitness else ""

    square_root_size_and_ratio: list[tuple[float, float]] = []

    list_avg_k: list[tuple[float, float]] = [0.0] * (size - 1)

    if False:
        max_k: int = 0
        min_k: int = 0

        for i in range(total_nodes_count):
            if i % 100 == 0:
                print(f"step={step}, i={i}")

            list_for_i: list[int] = list_k_i_by_time[i]

            if not list_for_i:
                list_for_i = [0] * size
                list_k_i_by_time[i] = list_for_i

            k_i: int = 0

            if i < len(list_nodes):
                node: Node = list_nodes[i]
                k_i = len(node.neighbors)

            step_index: int = step - 1

            list_for_i[step_index] = k_i

        for i in range(len(list_nodes)):
            node: Node = list_nodes[i]

            k_i: int = len(node.neighbors)

            max_k = max(max_k, k_i)
            min_k = k_i if min_k == 0 else min(min_k, k_i)

        ratio: float = max_k / min_k if min_k > 0 else 0.0
        square_root_size: float = math.sqrt(len(list_nodes))

        square_root_size_and_ratio.append((square_root_size, ratio))

        sum_k: int = sum(len(node.neighbors) for node in list_nodes)

        index_avg_k: int = step - 1

        list_avg_k[index_avg_k] = (sum_k / len(list_nodes), (2 * step * kernel_size) / len(list_nodes))

    plt.figure(figsize=(8, 6))

    for i in range(len(list_k_i_by_time)):
        if i % 100 == 0:
            print(f"plotting node {i} degree over time")
            list_degrees: list[int] = list_k_i_by_time[i]

            xs: list[int] = [0] * len(list_degrees)
            ys: list[int] = [0] * len(list_degrees)

            for t in range(len(list_degrees)):
                ki_t: int = list_degrees[t]

                xs[t] = t
                ys[t] = ki_t

            print(f"plotting node {i} degree over time")
            
            plt.loglog(xs, ys)

            print(f"plotted node {i} degree over time")
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Node Degree", fontsize=18)
    plt.title("Node degree over time in ba model")
    #plt.show()
    plt.savefig(f"ba_figs\\degree_over_time_loglog{file_name_fitness_part}.png")

    xs: list[int] = [i for i in range(1, size)]
    ys1: list[float] = [tup[0] for tup in list_avg_k]
    ys2: list[float] = [tup[1] for tup in list_avg_k]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys1, "-b")
    plt.plot(xs, ys2, "-r")
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Average Degree", fontsize=18)
    plt.title("average degree over time in ba model")
    #plt.show()
    plt.savefig(f"ba_figs\\average_degree_over_time{file_name_fitness_part}.png")

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
    plt.savefig(f"ba_figs\\ba_model_square_root_n_ratio{file_name_fitness_part}.png")

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

    degrees_count = sum(len(node.neighbors) for node in list_nodes)

    min_k = min(len(node.neighbors) for node in list_nodes)
    max_k = max(len(node.neighbors) for node in list_nodes)

    #for k in range(min_k, max_k + 1):
     #   count_k: int = 0
        
      #  if k in dict_degrees:
       #     count_k = dict_degrees[k]

        #    if count_k > 0:
         #       pk: float = count_k / degrees_count

          #      pks.append((k, pk))

    dict_degrees: dict[int, int] = {}
    
    for k in range(min_k, max_k + 1):
        count_k: int = 0
        
        for node in list_nodes:
            k_i: int = len(node.neighbors)

            if k_i == k:
                count_k += 1

        dict_degrees[k] = count_k

    list_degrees: list[int] = sorted(dict_degrees.keys())

    ks: list[int] = [0] * len(list_degrees)
    pks_values: list[float] = [0.0] * len(list_degrees)

    for i in range(len(list_degrees)):
        k: int = list_degrees[i]
        count_k: int = dict_degrees[k]

        ks[i] = k
        pks_values[i] = count_k / degrees_count

    plt.figure(figsize=(8, 6))

    #plt.plot(node_indices, ks, "-bD")
    plt.loglog(ks, pks_values, "-b")

    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)

    plt.title("ba model P(k)")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_p_k_loglog{file_name_fitness_part}.png")

    plt.figure(figsize=(8, 6))

    #plt.plot(node_indices, ks, "-bD")
    plt.plot(ks, pks_values, "-b")

    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)

    plt.title("ba model P(k)")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_p_k{file_name_fitness_part}.png")

    node_indices: list[int] = [0] * len(list_nodes)
    node_degrees: list[int] = [0] * len(list_nodes)
    calc_node_degrees: list[float] = [0] * len(list_nodes)

    max_k: int = 0
    min_k: int = 0

    for i in range(len(list_nodes)):
        node: Node = list_nodes[i]

        k_i: int = len(node.neighbors)

        max_k = max(max_k, k_i)
        min_k = k_i if min_k == 0 else min(min_k, k_i)

        node_indices[i] = i
        node_degrees[i] = k_i
        calc_node_degrees[i] = kernel_size * (len(list_nodes) / (i + 1)) ** 0.5

    diff: int = max_k - min_k

    n: int = 22

    ratio: float = max_k / min_k

    exp_n: float = 1/float(n)

    alpha: float = ratio ** exp_n

    print(f"alpha={alpha}")

    bins: list[tuple[float, int, list[Node]]] = [[]] * (n + 1)

    for i in range(n + 1):
        threshold: float = min_k * (alpha ** i)

        print(f"threshold[{i}]={threshold}")

        bins[i] = threshold, [], 0

    for node in list_nodes:
        k_i: int = len(node.neighbors)

        ratio_i: float = k_i / min_k

        log_ratio_i: float = math.log(ratio_i, alpha)

        index_bin: int = int(log_ratio_i)

        if index_bin < len(bins):
            bin: tuple[float, list[Node], int] = bins[index_bin]

            list_nodes: list[Node] = bin[1]

            list_nodes.append(node)
        else:
            print("no bin found!")
            return
        
    for i in range(len(bins)):
        bin: tuple[float, list[Node], int] = bins[i]
        list_nodes: list[Node] = bin[1]
        sorted_nodes: list[Node] = list(sorted(list_nodes, 
                           key=lambda node: len(node.neighbors)))
        
        if isinstance(sorted_nodes, list) and len(sorted_nodes) > 0:
            index_median: int = int(len(sorted_nodes) / 2)
            node_median: Node = sorted_nodes[index_median]

            bins[i] = bin[0], list_nodes, len(node_median.neighbors)

    bin_densities: dict[int, tuple[float, int]] = {}

    for i in range(len(bins)):
        bin: tuple[float, list[Node], int] = bins[i]
        next_bin: tuple[float, list[Node], int] = None

        if i < len(bins) - 1:
            next_bin = bins[i + 1]

        k_i: float = bin[0]
        k_next: float = next_bin[0] if next_bin is not None else min_k * (alpha ** (n + 1))

        list_nodes: list[Node] = bin[1]

        bin_n: int = len(list_nodes)

        width = k_next - k_i

        if bin_n > 0:
            density: float = bin_n / width
            bin_densities[i] = density, bin[2]

    print(f"len bin_densities={len(bin_densities)}, len list degrees={len(list_degrees)}")
        
    k_bins: list[tuple[int, float, int]] = []

    for i in range(len(list_degrees)):
        k: int = list_degrees[i]

        ratio_k: float = k / min_k

        log_ratio_k: float = math.log(ratio_k, alpha)

        bin_index: int = int(log_ratio_k)

        if bin_index in bin_densities.keys():
            tup: tuple[float, int] = bin_densities[bin_index]

            k_density: float = tup[0]
            k_median: int = tup[1]

            k_bins.append((k, k_density, k_median))
        else:
            print(f"No density for k={k} (bin_index={bin_index})")
            return

    plt.figure(figsize=(8, 6))

    xs: list[float] = [0] * len(k_bins)
    ys: list[float] = [0] * len(k_bins)

    for i in range(len(k_bins)):
        k_bin: tuple[int, int, float] = k_bins[i]

        k: int = k_bin[0]
        density: float = k_bin[1]

        xs[i] = k
        ys[i] = density

    print(f"len xs={len(xs)}, len ys={len(ys)}, len k bins={len(k_bins)}")

    plt.loglog(xs, ys, "-b")
    
    plt.xlabel("k", fontsize=18)
    plt.ylabel("density", fontsize=18)
    plt.title("ba model binned density")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_binned_density{file_name_fitness_part}.png")

    plt.figure(figsize=(8, 6))

    xs: list[float] = [0] * len(bin_densities)
    ys: list[float] = [0] * len(bin_densities)

    keys: list[int] = sorted(bin_densities.keys())

    for i in range(len(keys)):
        k: int = keys[i]

        tup: tuple[float, int] = bin_densities[k]

        median_k: int = tup[1]
        density: float = tup[0]

        xs[i] = median_k
        ys[i] = density

    print(f"len xs={len(xs)}, len ys={len(ys)}, len k bins={len(k_bins)}")

    plt.loglog(xs, ys, "-b")
    
    plt.xlabel("k", fontsize=18)
    plt.ylabel("density", fontsize=18)
    plt.title("ba model binned density (median k)")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_binned_density_median_k{file_name_fitness_part}.png")

    ratio: float = max_k / min_k if min_k > 0 else 0.0
    square_root_size: float = math.sqrt(len(list_nodes))
    
    print(f"size={len(list_nodes)}, ratio={ratio}, sqrt_size={square_root_size}")

    print(f"max_k={max_k}, min_k={min_k}")

    plt.figure(figsize=(8, 6))

    #plt.plot(node_indices, ks, "-bD")
    plt.plot(node_indices, node_degrees, "-b")
    plt.plot(node_indices, calc_node_degrees, "-r")

    plt.xlabel("node index (time)", fontsize=18)
    plt.ylabel("node degree", fontsize=18)

    plt.title("ba model degree by time")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_degree_by_time{file_name_fitness_part}.png")


def prepare_probabilities(list_nodes: list[Node], node_index: int, step: int) -> np.ndarray:
    total_links_weight: int = 0

    for i in range(node_index):
        node: Node = list_nodes[i]

        node_weight: int = len(node.neighbors)

        if node.fitness != 0.0:
            node_weight *= node.fitness

        total_links_weight += node_weight

    probabilities: list[float] = [0.0] * node_index

    for i in range(node_index):
        node: Node = list_nodes[i]

        node_weight: int = len(node.neighbors)

        if node.fitness != 0.0:
            node_weight *= node.fitness

        probabilities[i] = node_weight / total_links_weight

    return probabilities

def create_kernel(kernel_size: int) -> list[Node]:
    list_of_nodes: list[Node] = [None] * kernel_size

    for i in range(kernel_size):
        list_of_nodes[i] = Node(step=0)

    for i in range(kernel_size - 1):
        for j in range(i + 1, kernel_size):
            node_i: Node = list_of_nodes[i]
            node_j: Node = list_of_nodes[j]

            node_i.add_neighbor(node_j)
            node_j.add_neighbor(node_i)

    return list_of_nodes

num_steps: int = 2222

run_ba_model(num_steps=num_steps, kernel_size=4)
#run_ba_model(size=size, kernel_size=4, fitness=('exp', 5/size))
#run_ba_model(size=size, kernel_size=4, fitness=('mul', 10))
#run_ba_model(size=size, kernel_size=4, fitness=('pol', 10))