import math


def plot_gcc_graph(title="GCC Size vs Average Degree"):
    import matplotlib.pyplot as plt

    k_start: float = 0
    k_end: float = 6

    delta_k: float = 0.1

    points: list[tuple[float, float]] = []

    while k_start < k_end:
        k_value: float = k_start

        k_start += delta_k

        s_start: float = 0

        s_end: float = 1

        delta_s: float = 0.01

        tup_min = None

        while s_start < s_end:
            if k_value > 2:
                _ = 0
            s_value: float = s_start
            
            s_start += delta_s

            s_calculated: float = 1 - math.exp(k_value * s_value * -1)

            diff: float = abs(s_calculated - s_value)

            if s_value > 0 and (not isinstance(tup_min, tuple) or diff < tup_min[0]):
                tup_min = (diff, s_value, s_calculated)

        points.append((k_value, tup_min[1]))

        print(f"k={k_value:.2f}, s={tup_min[1]:.4f}, s_calculated={tup_min[2]:.4f}, diff={tup_min[0]:.6f}")

    plt.figure(figsize=(8, 6))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    plt.plot(xs, ys, "-bD")

    plt.xlabel("<k>", fontsize=18)
    plt.ylabel("s", fontsize=18)

    plt.title(title)
    #plt.show()
    plt.savefig("gcc_graph.png")

plot_gcc_graph()