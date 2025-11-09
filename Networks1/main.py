import numpy as np


def check_network(network: dict[int, list[int]]):
    list_keys = list(network.keys())

    key_index: int = 0

    found_error: bool = False

    while not found_error and key_index < len(list_keys):
        key: int = list_keys[key_index]
        key_index += 1

        links: list[int] = network[key]

        is_found: bool = False

        link_index: int = 0

        while not is_found and link_index < len(links):
            link: int = links[link_index]
            link_index += 1

            if link in network:
                link_list_of_links: list[int] = network[link]

                link_index_of_links: int = 0

                while not is_found and link_index_of_links < len(link_list_of_links):
                    link_from_list_of_links: int = link_list_of_links[link_index_of_links]
                    link_index_of_links += 1

                    is_found = link_from_list_of_links == key

        if not is_found:
            found_error = True

    return found_error


def create_matrix(network: dict[int, list[int]], should_print: bool = True):
    found_error: bool = check_network(network=network)

    if found_error:
        return

    indexed_network: dict[int, tuple[int, list[int]]] = {}

    index: int = 0

    for key in network:
        index += 1
        val: list[int] = network[key]

        indexed_network[key] = index, val

    dim: int = len(indexed_network)

    tup: tuple[int, list[bool]] = 0, []

    rows: list[tuple[int, list[bool]]] = [tup] * dim

    for key in indexed_network.keys():
        row: list[bool] = [False] * dim

        val: tuple[int, list[int]] = indexed_network[key]

        row_index: int = val[0]

        links: list[int] = val[1]

        for link in links:
            if link in indexed_network:
                val_link: tuple[int, list[int]] = indexed_network[link]

                col_index: int = val_link[0]

                if 0 < col_index <= dim:
                    row[col_index - 1] = True

        if 0 < row_index <= dim:
            tup = key, row
            rows[row_index - 1] = tup

    if should_print:
        str_print: str = "\\[\n\\hspace{-25mm}\n\\begin{bmatrix}\n"

        list_str_rows: list[str] = []

        str_row: str = " & ".join([f"\\bm{{{tup[0]}}}" for tup in rows])

        str_row = f"& {str_row}"

        list_str_rows.append(str_row)

        for tup in rows:
            row: list[bool] = tup[1]

            str_row = " & ".join(["1" if col else "0" for col in row])

            str_row = f"\\bm{{{tup[0]}}} & {str_row}"

            list_str_rows.append(str_row)

        str_print += "\\\\\n".join(list_str_rows)

        str_print += "\n\\end{bmatrix}\n\\]\n"

    return rows


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

    rows: list[tuple[int, list[bool]]] = create_matrix(network=network)

    matrix: list[list[int]] = []

    for tup_row in rows:
        cols: list[bool] = tup_row[1]

        list_cols: list[int] = [1 if col else 0 for col in cols]

        matrix.append(list_cols)

    dict_degrees: dict[int, int] = {}

    dim: int = len(matrix)

    for i in range(0, dim):
        tup: tuple[int, list[bool]] = rows[i]

        node: int = tup[0]

        cols: list[int] = matrix[i]

        degree: int = 0

        # \[k_i = \sum_{j=1}^{N}A_{ij}\]
        for j in range(0, dim):
            col: int = cols[j]

            degree += col

        dict_degrees[node] = degree

    average_degree: float = 0

    list_nodes: list[int] = list(dict_degrees.keys())

    # \[\langle{k}\rangle = \frac{1}{N}\sum_{i=1}^{N}k_i\]
    for i in range(0, dim):
        node: int = list_nodes[i]

        degree: int = dict_degrees[node]

        average_degree += degree

    average_degree /= dim

    dict_neighbor_degrees: dict[int, int] = {}

    dim: int = len(matrix)

    for i in range(0, dim):
        tup: tuple[int, list[bool]] = rows[i]

        node: int = tup[0]

        cols: list[int] = matrix[i]

        average_neighbors_degree: int = 0

        num_neighbors: int = 0

        # \[k_i,nn = \frac{1}{N}\sum_{j=1}^{N}A_{ij}k_j\]
        for j in range(0, dim):
            col: int = cols[j]

            num_neighbors += col

            list_neighbor_nodes: list[int] = matrix[j]

            neighbors_degree: int = 0

            # k_j = \sum_{h=1}^{N}A_{jh}
            for neighbor_col in list_neighbor_nodes:
                neighbors_degree += neighbor_col

            # A_{ij}k_j
            val: int = col * neighbors_degree

            # \sum_{j=1}^{N}A_{ij}k_j
            average_neighbors_degree += val

        # \[k_i,nn = \frac{\sum_{j=1}^{N}A_{ij}k_j}{\sum_{j=1}^{N}A_{ij}}\]
        average_neighbors_degree /= num_neighbors

        dict_neighbor_degrees[node] = average_neighbors_degree

    average_neighbors_degree: float = 0

    list_nodes: list[int] = list(dict_neighbor_degrees.keys())

    # \langle{k_{i,nn}}\rangle = \frac{1}{N}\sum_{i=1}^{N}k_{i,nn}
    for i in range(0, dim):
        node: int = list_nodes[i]

        degree: int = dict_neighbor_degrees[node]

        # \sum_{i=1}^{N}k_{i,nn}
        average_neighbors_degree += degree

    # \langle{k_{i,nn}}\rangle = \frac{1}{N}\sum_{i=1}^{N}k_{i,nn}
    average_neighbors_degree /= dim

    _ = 0

main()
