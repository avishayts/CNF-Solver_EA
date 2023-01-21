import matplotlib.pyplot as plt


def graph(results):
    plt.plot(results[0], results[1], 'blue')
    plt.axhline(y=0, color='green', linestyle='-')
    plt.xlabel("Generation #")
    plt.ylabel("Unsatisfiable clauses")
    plt.show()
