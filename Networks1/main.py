def create_matrix(network: dict[int, list[int]], should_print: bool = True):
    indexed_network: dict[int, tuple[int, list[int]]] = {}

    index: int = 0

    for key in network:
        index += 1
        val: list[int] = network[key]

        indexed_network[key] = index, val

    dim: int = len(indexed_network)

    tup: tuple[int, list[bool]] = 0, []

    rows: list[tuple[int, list[bool]]] = [tup] * dim

    for key in indexed_network:
        row: list[bool] = [False] * dim

        val: tuple[int, list[int]] = indexed_network[key]

        row_index: int = val[0]

        links: list[int] = val[1]

        for link in links:
            if link in indexed_network:
                val_link: tuple[int, list[int]] = indexed_network[link]

                col_index: int = val_link[0]

                if 0 <= col_index < dim:
                    row[col_index - 1] = True

        if 0 <= row_index < dim:
            tup = key, row
            rows[row_index - 1] = tup

    if should_print:
        str_print: str = "\\[\n\\begin{bmatrix}\n"

        list_str_rows: list[str] = []

        str_row: str = " & ".join([f"{tup[0]}" for tup in rows])

        str_row = f"& {str_row}"

        list_str_rows.append(str_row)

        for tup in rows:
            row: list[bool] = tup[1]

            str_row = " & ".join(["1" if col else "0" for col in row])

            str_row = f"{tup[0]} & {str_row}"

            list_str_rows.append(str_row)

        str_print += "\\\\\n".join(list_str_rows)

        str_print += "\\end{bmatrix}\n\\]\n"
    _ = 0


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

    create_matrix(network=network)


main()
