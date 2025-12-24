import math


def main():
    alpha = 0.5

    for n in range(3, 50000):
        mn: float = (n - 1) * math.exp(alpha * -1 * n / 2)
        mx: float = (n - 1) * math.exp(alpha * -1)

        print(f"n: {n}, min: {mn}, max: {mx}")

main()
