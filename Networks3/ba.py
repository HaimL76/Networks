from matplotlib import pyplot as plt
import numpy as np

class Node:
    def __init__(self):
        self.connections: list[Node] = []

    def add_connection(self, node: 'Node'):
        self.connections.append(node)


def main():
    counter: int = 0

    kernel_size: int = 4

    list_nodes: list[Node] = create_kernel(kernel_size)

    for step in range(1, 8520):
        new_node: Node = Node()

        list_indices: list[int] = [index for index in range(len(list_nodes))]

        probabilities = prepare_probabilities(list_nodes, kernel_size, step)

        selected_indices = np.random.choice(list_indices, replace=False, size=kernel_size, p=probabilities)

        for index in selected_indices:
            node: Node = list_nodes[index]

            node.add_connection(new_node)
            new_node.add_connection(node)

        list_nodes.append(new_node)

        list_degrees: list[str] = [str(len(node.connections)) for node in list_nodes]

        print(f"Step: {step}, nodes: {len(list_nodes)}")

    

    node_indices: list[int] = [0] * len(list_nodes)
    ks: list[int] = [0] * len(list_nodes)

    for i in range(len(list_nodes)):
        node: Node = list_nodes[i]

        node_indices[i] = i
        ks[i] = len(node.connections)

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
        links_count += len(node.connections)

    probabilities: list[float] = [0.0] * len(list_nodes)

    for i in range(len(list_nodes)):
        node: Node = list_nodes[i]
        probabilities[i] = len(node.connections) / links_count

    return probabilities

def create_kernel(kernel_size: int) -> list[list[Node]]:
    list_of_nodes: list[Node] = [None] * kernel_size

    for i in range(kernel_size):
        list_of_nodes[i] = Node()

    for i in range(kernel_size - 1):
        for j in range(i + 1, kernel_size):
            node_i: Node = list_of_nodes[i]
            node_j: Node = list_of_nodes[j]

            node_i.add_connection(node_j)
            node_j.add_connection(node_i)

    return list_of_nodes

main()