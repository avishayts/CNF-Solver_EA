import re
from matplotlib import pyplot as plt


def get_data_sudoku(statistics_file_name, num_of_clauses):
    with open(statistics_file_name) as f:
        generations = []
        unsatisfiable_clauses = []
        generation, best_fitness = None, None
        for line in f:
            if line.startswith("generation"):
                match = re.search(r'\d+', line)
                if match:
                    generation = int(match.group())
                generations.append(generation)
            elif line.startswith("best fitness"):
                best_fitness = float(line.split()[-1])
                unsatisfiable_clauses.append(num_of_clauses - best_fitness)
    return generations, unsatisfiable_clauses


def plot_graphs_sudoku(data):
    x, y = data
    plt.plot(x, y, color='blue')
    plt.xlabel("Generation #")
    plt.ylabel("Unsatisfiable clauses")
    plt.axhline(color='g', linestyle='--')
    plt.show()


def show_graph_sudoku(output_file, num_of_clauses):
    data = get_data_sudoku(output_file, num_of_clauses)
    plot_graphs_sudoku(data)
