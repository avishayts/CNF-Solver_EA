import re
from matplotlib import pyplot as plt


def get_data(statistics_file_name, num_of_clauses):
    with open(statistics_file_name) as f:
        generations = []
        unsasifiable_clauses = []
        generation, best_fitness = None, None
        for line in f:
            if line.startswith("generation"):
                match = re.search(r'\d+', line)
                if match:
                    generation = int(match.group())
                generations.append(generation)
            elif line.startswith("best fitness"):
                best_fitness = float(line.split()[-1])
                unsasifiable_clauses.append(num_of_clauses - best_fitness)
    return generations, unsasifiable_clauses


def plot_graphs(data):
    x, y = data
    plt.plot(x, y, color='blue')
    plt.xlabel("Generation #")
    plt.ylabel("Unsatisfiable clauses")
    plt.axhline(color='g', linestyle='--')
    plt.show()


def show_graph(output_file, num_of_clauses):
    data = get_data(output_file, num_of_clauses)
    plot_graphs(data)
