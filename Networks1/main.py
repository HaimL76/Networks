def create_matrix(network: dict[int, list[int]]):
    indexed_network: dict[int, tuple[int, list[int]]] = {}

    index: int = 0

    for key in network:
        index += 1
        val: list[int] = network[key]

        indexed_network[key] = index, val

    links_matrix

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
