import numpy as np


def check_network(network: dict[int, list[int]]):
    nodes = list(network.keys())

    node_index: int = 0

    found_error: bool = False

    while not found_error and node_index < len(nodes):
        node: int = nodes[node_index]
        node_index += 1

        neighbors: list[int] = network[node]

        is_found: bool = False

        neighbor_index: int = 0

        while not is_found and neighbor_index < len(neighbors):
            neighbor: int = neighbors[neighbor_index]
            neighbor_index += 1
            
            is_found = neighbor == node

        if is_found:
            found_error = True
            break

        is_found = False

        neighbor_index = 0

        while not is_found and neighbor_index < len(neighbors):
            neighbor: int = neighbors[neighbor_index]
            neighbor_index += 1

            if neighbor in network:
                neighbor_neighbors: list[int] = network[neighbor]

                neighbor_neighbors_index: int = 0

                while not is_found and neighbor_neighbors_index < len(neighbor_neighbors):
                    link_from_list_of_links: int = neighbor_neighbors[neighbor_neighbors_index]
                    neighbor_neighbors_index += 1
                    is_found = link_from_list_of_links == node

        if not is_found:
            found_error = True

    return found_error


def create_matrix(network: dict[int, list[int]], should_print: bool = True):
    found_error: bool = check_network(network=network)

    if found_error:
        return
    
    nodes: list[int] = list(network.keys())

    dim: int = len(nodes)

    rows: list[list[bool]] = [[]] * dim

    for i in range(dim):
        rows[i] = [False] * dim

    for i in range(dim - 1):
        node_i: int = nodes[i]

        neighbors: list[int] = network[node_i]

        for j in range(i + 1, dim):
            node_j: int = nodes[j]

            rows[i][j] = rows[j][i] = node_j in neighbors

    if should_print:
        str_print: str = "\\[\\{" + ",".join([f"\\overset{{{i+1}}}{{{nodes[i]}}}" for i in range(dim)]) + "\\}\\]"

        str_print += "\\[\n\\hspace{-25mm}\n\\begin{bmatrix}\n"

        list_str_rows: list[str] = []

        str_row: str = " & ".join([f"\\bm{{{node}}}" for node in nodes])

        str_row = f"& {str_row}"

        list_str_rows.append(str_row)

        for i in range(dim):
            node: int = nodes[i]

            row: list[bool] = rows[i]

            str_row = " & ".join(["1" if col else "0" for col in row])

            str_row = f"\\bm{{{node}}} & {str_row}"

            list_str_rows.append(str_row)

        str_print += "\\\\\n".join(list_str_rows)

        str_print += "\n\\end{bmatrix}\n\\]\n"

        with open(r"neighbors_matrix.txt", "w") as fw:
            fw.write(str_print)

    return nodes, rows

def calculate_degrees(nodes: list[int], rows: list[list[bool]], should_print: bool = False):
    if not isinstance(rows, list):
        return

    matrix: list[list[int]] = []

    for row in rows:
        list_cols: list[int] = [1 if col else 0 for col in row]

        matrix.append(list_cols)

    degrees: dict[int, int] = {}

    dim: int = len(matrix)

    for i in range(dim):
        node: int = nodes[i]

        row: list[int] = matrix[i]

        degree: int = 0

        # \[k_i = \sum_{j=1}^{N}A_{ij}\]
        for j in range(dim):
            a_ij: int = row[j]

            degree += a_ij

        degrees[node] = degree

    average_degree: float = 0

    # \[\langle{k}\rangle = \frac{1}{N}\sum_{i=1}^{N}k_i\]
    for i in range(dim):
        node: int = nodes[i]

        degree: int = degrees[node]

        average_degree += degree

    average_degree /= dim

    neighbor_degrees: dict[int, int] = {}

    for i in range(dim):
        node: int = nodes[i]

        row_i: list[int] = matrix[i]

        degree: int = 0

        neighbors_count: int = 0

        for j in range(dim):
            neighbor: int = nodes[j]

            row_j: list[int] = matrix[j]

            a_ij: int = row_i[j]

            neighbors_count += a_ij

            if neighbors_count > 0:
                for h in range(dim):
                    a_jh: int = row_j[h]

                    degree += a_ij * a_jh
        
        neighbor_degrees[node] = degree / neighbors_count if neighbors_count > 0 else 0

    average_neighbors_degree: float = 0

    # \[\langle{k}\rangle = \frac{1}{N}\sum_{i=1}^{N}k_i\]
    for i in range(dim):
        node: int = nodes[i]

        degree: int = neighbor_degrees[node]

        average_neighbors_degree += degree

    average_neighbors_degree /= dim

    str_print: str = "\\[\\begin{matrix}\n"
    str_print += "\\\\\n".join([f"k_{{{i+1}}}={degrees[nodes[i]]}" for i in range(dim)])
    str_print += "\\end{matrix}\\]\n"

    str_print += f"\\[\\langle{{k}}\\rangle={average_degree}\\]\n"

    str_print += "\\[\\begin{matrix}\n"
    str_print += "\\\\\n".join([f"k_{{{i+1},nn}}={neighbor_degrees[nodes[i]]}" for i in range(dim)])
    str_print += "\\end{matrix}\\]\n"

    str_print += f"\\[\\langle{{k,nn}}\\rangle={average_neighbors_degree}\\]\n"

    with open(r"degrees.txt", "w") as fw:
        fw.write(str_print)

    return degrees, average_degree, neighbor_degrees, average_neighbors_degree

def calculate_lengths(network: dict[int, list[int]], 
                      rows: list[tuple[int, list[bool]]],
                      should_print: bool = False):
    nodes: list[int] = list(network.keys())
    
    neighbors_matrix: list[list[int]] = []

    for row in rows:
        list_cols: list[int] = [1 if col else 0 for col in row]

        neighbors_matrix.append(list_cols)

    np_neighbors_matrix: np.ndarray = np.array(neighbors_matrix)

    dim: int = len(nodes)

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

    if should_print:
        str_print: str = "\\[\n\\hspace{-25mm}\n\\begin{bmatrix}\n"

        list_str_rows: list[str] = []

        str_row: str = " & ".join([f"\\bm{{{node}}}" for node in nodes])

        str_row = f"& {str_row}"

        list_str_rows.append(str_row)

        for i in range(dim):
            node: int = nodes[i]

            row: list[int] = lengths[i]

            str_row = " & ".join([str(col) for col in row])

            str_row = f"\\bm{{{node}}} & {str_row}"

            list_str_rows.append(str_row)

        str_print += "\\\\\n".join(list_str_rows)

        str_print += "\n\\end{bmatrix}\n\\]\n"

        str_print += f"\\[\\langle{{length}}\\rangle={average_length}\\]\n"

        with open(r"lengths.txt", "w") as fw:
            fw.write(str_print)

    return lengths, average_length

def count_triangles(network: dict[int, list[int]], rows: list[tuple[int, list[bool]]]):
    nodes: list[int] = list(network.keys())
    
    neighbors_matrix: list[list[int]] = []

    for row in rows:
        list_cols: list[int] = [1 if col else 0 for col in row]

        neighbors_matrix.append(list_cols)

    np_neighbors_matrix: np.ndarray = np.array(neighbors_matrix)

    dim: int = len(nodes)

    np_lengths_matrix: np.ndarray = np.identity(dim, dtype=int)

    lengths: list[list[int]] = [[]] * dim

    for i in range(dim):
        lengths[i] = [0] * dim

    for l in range(3):
        np_lengths_matrix = np.matmul(np_lengths_matrix, np_neighbors_matrix)

    nodes_in_triangles: dict[int, int] = {}
    
    triangles_count: int = 0

    for i in range(dim):
        node: int = nodes[i]

        val: int = int(np_lengths_matrix[i][i])

        triangles_count += val

        if val > 0:
            nodes_in_triangles[node] = val

    return triangles_count / 6

def main():
    network: dict[int, list[int]] = {
        426: [345, 365, 245, 121, 165, 782, 452],
        345: [426, 365, 153, 245],
        365: [426, 345],
        153: [345],
        245: [345, 426],
        165: [426, 358, 369],
        358: [165, 452, 546],
        121: [426, 143, 131, 782],
        452: [426, 358, 272],
        143: [121],
        131: [121],
        272: [452, 782, 171],
        546: [358],
        369: [165],
        171: [272],
        782: [272, 426, 121, 888],
        888: [782]
    }

    nodes, rows = create_matrix(network=network)

    tup: tuple = calculate_degrees(nodes=nodes, rows=rows)

    if isinstance(tup, tuple) and len(tup) == 4:
        dict_degrees, average_degree, dict_neighbor_degrees, average_neighbors_degree = tup

        print("Degrees:", dict_degrees)
        print("Average Degree:", average_degree)
        print("Neighbor Degrees:", dict_neighbor_degrees)
        print("Average Neighbor Degree:", average_neighbors_degree)
    else:
        print("Error in calculating degrees.")

    tup: tuple = calculate_lengths(network=network, rows=rows, should_print=True)

    if isinstance(tup, tuple) and len(tup) == 2:
        lengths, average_length = tup

        print("lengths:", lengths)
        print("average_length:", average_length)
    else:
        print("Error in calculating lengths.")

    triangles_count: int = count_triangles(network=network, rows=rows)

    print("triangles_count:", triangles_count)

main()
