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

fitness_none: int = 0
fitness_linear: int = fitness_none + 1
fitness_square: int = fitness_linear + 1

def get_str_fitness(fitness: int) -> str:
    str_fitness: str = ""

    if fitness == fitness_linear:
        str_fitness = "_fitness_linear"
    elif fitness == fitness_square:
        str_fitness = "_fitness_square"

    return str_fitness

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

def run_ba_model(num_steps: int, kernel_size: int, fitness: int = fitness_none):
    os.makedirs("ba_figs", exist_ok=True)

    fitness_value: float = 0.0

    if fitness > fitness_none:
        fitness_value = 1.0

    kernel: list[Node] = create_kernel(kernel_size, fitness=fitness_value)

    node_index: int = len(kernel)

    nodes_count: int = node_index + num_steps

    list_nodes: list[Node]  = [None] * nodes_count

    list_nodes[:node_index] = kernel

    num_steps_to_track: int = int(math.log(nodes_count))
    
    ki_by_time: list[tuple[int, list[int]]] = [None] * num_steps_to_track

    num_list_nodes: int = len(list_nodes)

    diff: int = int(num_list_nodes / num_steps_to_track)

    for i in range(num_steps_to_track):
        ti: int = int(np.e ** i)
        if ti == 0:
            ti = 1

        ki_by_time[i] = (ti, [])

    k: int = kernel_size - 1

    k_max: int = k

    k_ratio_n: list[tuple[float, float]] = [None] * nodes_count
    k_average: list[float] = [0.0] * nodes_count

    k_average_kernel: float = k

    k_ratio_n[:node_index] = [(node_index, 1) for _ in kernel]
    k_average[:node_index] = [k_average_kernel for _ in kernel]

    calculated_degrees: list[list[int]] = None

    if fitness > fitness_none:
        calculated_degrees = [[]] * (num_steps + 1)

        initial_degree: list[int] = calculated_degrees[0]

        initial_degree = [0] * len(kernel)

        for i in range(kernel_size):
            node: Node = list_nodes[i]

            initial_degree[i] = len(node.neighbors)

        calculated_degrees[0] = initial_degree

    for step in range(num_steps):
        curr_node_index: int = node_index
        node_index += 1

        step_id: int = step + 1

        fitness_value: float = 0.0

        if fitness == fitness_linear:
            fitness_value = step_id
        elif fitness == fitness_square:
            fitness_value = step_id ** 2

        new_node: Node = Node(step=step_id, fitness=fitness_value)

        list_nodes[curr_node_index] = new_node

        probabilities = prepare_probabilities(list_nodes, node_index=node_index)
        
        list_indices: list[int] = [index for index in range(len(probabilities))]

        selected_indices = np.random.choice(list_indices, replace=False, size=kernel_size, p=probabilities)

        if has_double(selected_indices):
            raise Exception("double selection of nodes")

        for index in selected_indices:
            node: Node = list_nodes[index]

            num_neighbors_before_add: int = len(node.neighbors) + len(new_node.neighbors)

            node.add_neighbor(new_node)
            new_node.add_neighbor(node)

            num_neighbors_after_add: int = len(node.neighbors) + len(new_node.neighbors)

            num_added_neighbors: int = num_neighbors_after_add - num_neighbors_before_add

            k += num_added_neighbors

            node_k: int = len(node.neighbors)

            k_max: int = max(k_max, node_k)

        new_node_k: int = len(new_node.neighbors)

        k_max: int = max(k_max, new_node_k)

        k_min: int = k_max

        for i in range(node_index):
            node: Node = list_nodes[i]
            k_min: int = min(k_min, len(node.neighbors))
        
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
        
        if fitness > fitness_none:
            m: int = len(kernel)

            index: int = step + 1

            curr_calculated_degrees: list[int] = [0] * (len(kernel) + index)

            calculated_degrees[index] = curr_calculated_degrees

            prev_calculated_degrees: list[int] = calculated_degrees[index - 1]

            sum_prev: float = 0.0

            for j in range(len(prev_calculated_degrees)):
                k_j: int = prev_calculated_degrees[j]

                node: Node = list_nodes[j]

                f_j: float = node.fitness

                degree_fitness: float = k_j * f_j

                curr_calculated_degrees[j] = degree_fitness

                sum_prev += degree_fitness

            j: int = 0

            while j < len(prev_calculated_degrees):
                k_j: int = prev_calculated_degrees[j]

                degree_fitness: float = curr_calculated_degrees[j]

                p_j: float = degree_fitness / sum_prev
                
                curr_k_j = k_j + m * p_j

                curr_calculated_degrees[j] = curr_k_j
                
                j += 1

            curr_calculated_degrees[j] = len(kernel)

    save_ki_by_time_plot(ki_by_time=ki_by_time, calculated_degrees=calculated_degrees,
                         kernel_size=kernel_size, fitness=fitness)

    save_k_average_n_plot(kernel_size=kernel_size, k_average=k_average, 
                          fitness=fitness)
    save_k_ratio_n_plot(k_ratio_n=k_ratio_n, fitness=fitness)

    dict_k: dict[int, int] = {}

    for node in list_nodes:
        k_i: int = len(node.neighbors)

        if k_i not in dict_k:
            dict_k[k_i] = 0

        dict_k[k_i] += 1

    save_p_k_plot(dict_k=dict_k, kernel_size=kernel_size,
                  nodes_count=nodes_count, fitness=fitness)
    
    save_p_k_plot(dict_k=dict_k, kernel_size=kernel_size,
                  nodes_count=nodes_count, fitness=fitness,
                  with_calculated_slope=True)
    
    n_max: int = 15
    
    save_p_k_plot_log_binning(list_nodes=list_nodes, n_max=n_max, dict_k=dict_k,
                              kernel_size=kernel_size, fitness=fitness)

    save_p_k_plot_log_binning(list_nodes=list_nodes, n_max=n_max, dict_k=dict_k,
                              kernel_size=kernel_size, fitness=fitness, 
                              with_calculated_slope=True)
    
    save_p_k_plot_log_binning(list_nodes=list_nodes, n_max=n_max, dict_k=dict_k,
                              kernel_size=kernel_size, fitness=fitness, 
                              take_bins_medians=True)

    save_p_k_plot_log_binning(list_nodes=list_nodes, n_max=n_max, dict_k=dict_k,
                              kernel_size=kernel_size, fitness=fitness, 
                              with_calculated_slope=True, take_bins_medians=True)
    
def save_ki_by_time_plot(ki_by_time: list[tuple[int, list[int]]],
                         calculated_degrees: list[list[int]],
                        kernel_size: int, fitness: int = fitness_none):
    str_fitness: str = get_str_fitness(fitness)
    
    # Rainbow colors list
    rainbow_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    
    num_steps: int = len(ki_by_time[0][1])

    xs: list[int] = [0.0] * num_steps
    list_ys: list[list[tuple[float, float]]] = [[]] * num_steps

    plt.figure(figsize=(8, 6))
    plt.xlabel("t", fontsize=18)
    plt.ylabel("ki", fontsize=18)
    plt.title("ba model ki by t")

    for i in range(len(ki_by_time)):
        tup: tuple[int, list[int]] = ki_by_time[i]

        ti: int = tup[0]
        list_ki: list[int] = tup[1]

        ys_ki: list[float] = [0.0] * len(list_ki)
        ys_calc_ki: list[float] = [0.0] * len(list_ki)

        for t in range(len(list_ki)):
            xs[t] = t

            calc_ki: float = 0.0

            take_from_calculated: bool = False

            if isinstance(calculated_degrees, list) and t < len(calculated_degrees):
                degrees_t: list[int] = calculated_degrees[t]
                
                if isinstance(degrees_t, list) and ti < len(degrees_t):
                    calc_ki = degrees_t[ti]
                    take_from_calculated = True

            if not take_from_calculated:
                calc_ki = kernel_size * ((t/ti) ** 0.5)
            
            ki: int = list_ki[t]

            ys_ki[t] = ki
            ys_calc_ki[t] = calc_ki

        color = rainbow_colors[i % len(rainbow_colors)]

        plt.loglog(xs, ys_ki, color=color)
        plt.loglog(xs, ys_calc_ki, color=color)

    plt.ylim(bottom=kernel_size - 1)
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_ki_t_loglog{str_fitness}.png")
    plt.close()

def save_p_k_plot_log_binning(list_nodes: list[Node], n_max: int,
        dict_k: dict[int, int], kernel_size: int = 0, fitness: int = fitness_none,
        with_calculated_slope: bool = False,
        take_bins_medians: bool = False):
    str_fitness: str = get_str_fitness(fitness)

    ks: list[int] = sorted(dict_k.keys())

    sum_k: int = sum(ks)

    k_min: int = ks[0]
    k_max: int = ks[-1]

    alpha: float = (k_max / k_min) ** (1 / n_max)

    num_bins: int = n_max

    list_bins: list[tuple[float, list[Node]]] = [None] * num_bins

    for node in list_nodes:
        k_i: int = len(node.neighbors)

        bin_index: int = None

        index: int = 0

        while bin_index is None and index < len(list_bins):
            curr_index: int = index
            index += 1
            
            bin: tuple[float, list[Node]] = list_bins[curr_index]

            if bin is None:
                list_bins[curr_index] = (None, None)

            bin = list_bins[curr_index]

            k_start: float = bin[0]
            
            if k_start is None:
                k_start = k_min * (alpha ** curr_index)
                list_bins[curr_index] = (k_start, None)

            k_end: float = k_start * alpha

            if k_i >= k_start and k_i < k_end:
                bin_index = curr_index

        #print(f"node k_i={k_i}, bin_index={bin_index}")

        if bin_index is None:
            bin_index = -1

        bin: tuple[float, list[Node]] = list_bins[bin_index]

        if isinstance(bin, tuple) and len(bin) == 2:
            k_start: float = bin[0]
            bin_nodes: list[Node] = bin[1]

            if not isinstance(bin_nodes, list):
                bin_nodes = []
                list_bins[bin_index] = (k_start, bin_nodes)

            bin_nodes.append(node)

    list_bin_densities: list[float] = [0.0] * len(list_bins)

    for index in range(len(list_bins)):
        bin: tuple[float, list[Node]] = list_bins[index]

        k_min: float = bin[0]

        k_max: float = None

        index_next: int = index + 1

        if index_next < len(list_bins):
            next_bin = list_bins[index_next]
            k_max = next_bin[0]

        if k_max is None:
            k_max = k_min * alpha

        bin_nodes: list[Node] = bin[1]

        num_bin_nodes: int = 0

        if isinstance(bin_nodes, list):
            num_bin_nodes = len(bin_nodes)

        bin_size: float = k_max - k_min

        density: float = num_bin_nodes / bin_size
        
        list_bin_densities[index] = density / sum_k

    dict_k_bins: dict[int, float] = {}

    for i in range(len(ks)):
        k: int = ks[i]
        
        bin_index: int = None

        index: int = 0

        while bin_index is None and index < len(list_bins):
            curr_index: int = index
            index += 1
            
            bin: tuple[float, list[Node]] = list_bins[curr_index]

            k_start: float = bin[0]
            k_end: float = k_start * alpha

            if k >= k_start and k < k_end:
                bin_index = curr_index

        if bin_index is None:
            bin_index = -1

        dict_k_bins[k] = list_bin_densities[bin_index]

    len_arr: int = len(dict_k_bins)

    if take_bins_medians:
        len_arr = len(list_bins)

    xs: list[int] = [0] * len_arr
    ys: list[float] = [0.0] * len_arr

    if with_calculated_slope:
        ys_calc: list[float] = [0.0] * len_arr

    if take_bins_medians:
        for i in range(len(list_bins)):
            k_mean: float = 0.0

            bin: tuple[float, list[Node]] = list_bins[i]

            if isinstance(bin, tuple) and len(bin) == 2:
                bin_nodes: list[Node] = bin[1]
                
                if isinstance(bin_nodes, list) and len(bin_nodes) > 0:
                    bin_nodes = list(sorted(bin_nodes, 
                                       key=lambda node: len(node.neighbors)))

                if isinstance(bin_nodes, list) and len(bin_nodes) > 0:
                    index_mean: int = int(len(bin_nodes) / 2)
                    node_mean: Node = bin_nodes[index_mean]

                    k_mean = len(node_mean.neighbors)

            density_k: float = list_bin_densities[i]

            xs[i] = k_mean
            ys[i] = density_k

            if with_calculated_slope:
                calculated_y: float = 0.0

                if k_mean > 0:
                    calculated_y = 2 * (kernel_size ** 2) * (k_mean ** -3)

                ys_calc[i] = calculated_y
    else:
        ks: list[int] = sorted(dict_k_bins.keys())

        for i in range(len(ks)):
            k: int = ks[i]
            density_k: float = dict_k_bins[k]

            xs[i] = k
            ys[i] = density_k

            if with_calculated_slope:
                calculated_y: float = 0.0

                if k > 0:
                    calculated_y = 2 * (kernel_size ** 2) * (k ** -3)

                ys_calc[i] = calculated_y

    plt.figure(figsize=(8, 6))
    plt.loglog(xs, ys, "-b")

    if with_calculated_slope:
        plt.loglog(xs, ys_calc, "-r")

    plt.xlim(left=ks[0])
    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)
    plt.title("ba model P(k)")
    #plt.show()
    str_with_calculated_slope: str = ""
    
    if with_calculated_slope:
        str_with_calculated_slope = "_with_slope"

    str_take_bins_medians: str = ""

    if take_bins_medians:
        str_take_bins_medians = "_take_medians"

    plt.savefig(f"ba_figs\\ba_model_p_k_loglog_binning{str_with_calculated_slope}{str_take_bins_medians}{str_fitness}.png")
    plt.close()

def save_p_k_plot(dict_k: dict[int, int], kernel_size: int, 
                  nodes_count: int, fitness: int = fitness_none,
                  with_calculated_slope: bool = False):
    str_fitness: str = "_with_fitness" if fitness else ""

    ks: list[int] = sorted(dict_k.keys())

    xs: list[int] = [0] * len(ks)
    ys: list[float] = [0.0] * len(ks)

    if with_calculated_slope:
        ys1: list[int] = [0] * len(ks)

    b: float = None

    for i in range(len(ks)):
        k: int = ks[i]
        count_k: int = dict_k[k]

        xs[i] = k
        ys[i] = count_k / nodes_count

        if with_calculated_slope:
            ys1[i] = 2 * (kernel_size ** 2) * (k ** -3)

    plt.figure(figsize=(8, 6))
    plt.loglog(xs, ys, "-b")
    if with_calculated_slope:
        plt.loglog(xs, ys1, "-r")
    # Add xticks by powers of e
    e_powers = [np.exp(i) for i in range(int(np.log(max(xs))) + 1)]
    arr_x: list[str] = [f'$e^{{{i}}}$' for i in range(len(e_powers))]
    arr_y: list[str] = [f'$e^{{{i*-1}}}$' for i in range(len(e_powers))]
    #plt.xticks(e_powers, arr_x)
    #plt.yticks(e_powers, arr_y)
    plt.xlim(left=kernel_size - 1)
    plt.xlabel("k", fontsize=18)
    plt.ylabel("P(k)", fontsize=18)
    plt.title("ba model P(k)")
    #plt.show()
    str_with_calculated_slope: str = ""
    
    if with_calculated_slope:
        str_with_calculated_slope = "_with_slope"
    
    plt.savefig(f"ba_figs\\ba_model_p_k_loglog{str_with_calculated_slope}{str_fitness}.png")
    plt.close()

def save_k_average_n_plot(kernel_size: int, k_average: list[float], 
                          fitness: int = fitness_none):
    str_fitness: str = get_str_fitness(fitness)

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
    plt.savefig(f"ba_figs\\ba_model_k_average_n{str_fitness}.png")
    plt.close()

def save_k_ratio_n_plot(k_ratio_n: list[tuple[float, float]], fitness: bool):
    str_fitness: str = "_with_fitness" if fitness else ""

    xs: list[int] = [0] * len(k_ratio_n)
    ys: list[float] = [0.0] * len(k_ratio_n)

    for i in range(len(k_ratio_n)):
        tup: tuple[float, float] = k_ratio_n[i]
        xs[i] = tup[0]
        ys[i] = tup[1]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, "-b")
    plt.xlabel("Square root of N", fontsize=18)
    plt.ylabel("Max and Min Degrees Ratio", fontsize=18)
    plt.title("ba model max and min degrees ratio by square root of N")
    #plt.show()
    plt.savefig(f"ba_figs\\ba_model_k_ratio_n{str_fitness}.png")
    plt.close()

def prepare_probabilities(list_nodes: list[Node], node_index: int) -> np.ndarray:
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

def create_kernel(kernel_size: int, fitness: float = 0.0) -> list[Node]:
    list_of_nodes: list[Node] = [None] * kernel_size

    for i in range(kernel_size):
        list_of_nodes[i] = Node(step=0, fitness=fitness)

    for i in range(kernel_size - 1):
        for j in range(i + 1, kernel_size):
            node_i: Node = list_of_nodes[i]
            node_j: Node = list_of_nodes[j]

            node_i.add_neighbor(node_j)
            node_j.add_neighbor(node_i)

    return list_of_nodes

num_steps: int = 22222

run_ba_model(num_steps=num_steps, kernel_size=4)
run_ba_model(num_steps=num_steps, kernel_size=4, fitness=fitness_linear)
run_ba_model(num_steps=num_steps, kernel_size=4, fitness=fitness_square)