import re
from matplotlib import pyplot as plt
import numpy as np

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


# All the results was found in prior experiment and search.
def compare_results():
    a = [(1.8250384330749512, 1.3332860469818115),
         (2.7977333068847656, 2.5233566761016846),
         (3.224578619003296, 3.1564507484436035),
         (3.834014654159546, 3.9264132976531982)]
    b = [(0.08701086044311523, 0.07080960273742676),
         (2.4873955249786377, 0.5346722602844238),
         (1.7066280841827393, 3.138984441757202),
         (3.1067240238189697, 21.88584041595459)]
    c = [(0.0010390281677246094, 0.0010390281677246094),
         (0.0, 0.0),
         (0.001013040542602539, 0.001013040542602539),
         (0.0010097026824951172, 0.0010097026824951172)]
    d = [(0.01300668716430664, 10), (10, 10), (10, 10), (10, 10)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in zip([0, 1, 2], axes):
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in a]), label="default")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in b]), label="improved")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in c]), label="pysat")
        # ax.plot(np.array([i*100 for i in range(1, 6)]), np.array([i for i in d]), label="naive")
        ax.set_xlabel('N')
        ax.set_ylabel('runtime')
        ax.legend()
    plt.show()

    a = [(1.8250384330749512, 1.3332860469818115),
         (2.7977333068847656, 2.5233566761016846),
         (3.224578619003296, 3.1564507484436035),
         (3.607895612716675, 4.230013847351074)]
    b = [(0.08701086044311523, 0.07080960273742676),
         (2.4873955249786377, 0.5346722602844238),
         (1.7066280841827393, 3.138984441757202),
         (2.7316837310791016, 2.9303078651428223)]
    c = [(0.0010390281677246094, 0.0010390281677246094),
         (0.0, 0.0),
         (0.001013040542602539, 0.001013040542602539),
         (0.0010097026824951172, 0.0010097026824951172)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in zip([0, 1, 2], axes):
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in a]), label="default")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in b]), label="improved")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in c]), label="pysat")
        ax.set_xlabel('N')
        ax.set_ylabel('runtime')
        ax.legend()
    plt.show()

    a = [(1.8250384330749512, 1.3332860469818115),
         (2.7977333068847656, 2.5233566761016846),
         (3.224578619003296, 3.1564507484436035),
         (3.607895612716675, 4.230013847351074)]
    b = [(0.08701086044311523, 0.07080960273742676),
         (2.4873955249786377, 0.5346722602844238),
         (1.7066280841827393, 3.138984441757202),
         (2.7316837310791016, 2.9303078651428223)]
    c = [(0.0010390281677246094, 0.0010390281677246094),
         (0.0, 0.0),
         (0.001013040542602539, 0.001013040542602539),
         (0.0010097026824951172, 0.0010097026824951172)]
    d = [(0.01300668716430664, 10), (10, 10), (10, 10), (10, 10)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in zip([0, 1], axes):
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in a]), label="default")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in b]), label="improved")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in c]), label="pysat")
        ax.plot(np.array([i * 25 for i in range(1, 5)]), np.array([j[i] for j in d]), label="naive")
        ax.set_xlabel('N')
        ax.set_ylabel('runtime')
        ax.legend()
    plt.show()

    a = [(98, 100), (198, 197), (292, 295), (393, 392)]
    b = [(100, 100), (198, 200), (300, 295), (400, 400)]
    where_default = [i for i in range(17, 118, 25)]
    where_improved = [i for i in range(27, 128, 25)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in zip([0, 1], axes):
        ax.bar([20.0, 45.0, 70.0, 95.0], [j[i] for j in a], width=10, color='blue', label="default")
        ax.bar([30.0, 55.0, 80.0, 105.0], [j[i] for j in b], width=10, color='orange', label="improved")
        ax.set(xticks=np.arange(25, 125, 25), yticks=np.arange(100, 450, 50))
        ax.set_xlabel('N')
        ax.set_ylabel('satisfied clauses')
        for j, val in enumerate(a):
            ax.text(where_default[j], 40 + 40 * j, str(val[i]), color="black", fontweight="bold")
        for j, val in enumerate(b):
            ax.text(where_improved[j], 40 + 40 * j, str(val[i]), color="white", fontweight="bold")
        ax.legend()
    plt.show()

    a = [(100, 99), (187, 187), (275, 275), (360, 360)]
    b = [(100, 100), (198, 200), (300, 295), (400, 400)]
    where_default = [i for i in range(17, 118, 25)]
    where_improved = [i for i in range(27, 128, 25)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in zip([0, 1], axes):
        ax.bar([20.0, 45.0, 70.0, 95.0], [j[i] for j in a], width=10, color='red', label="naive")
        ax.bar([30.0, 55.0, 80.0, 105.0], [j[i] for j in b], width=10, color='orange', label="improved")
        ax.set(xticks=np.arange(25, 125, 25), yticks=np.arange(100, 450, 50))
        ax.set_xlabel('N')
        ax.set_ylabel('satisfied clauses')
        for j, val in enumerate(a):
            ax.text(where_default[j], 40 + 40 * j, str(val[i]), color="black", fontweight="bold")
        for j, val in enumerate(b):
            ax.text(where_improved[j], 40 + 40 * j, str(val[i]), color="white", fontweight="bold")
        ax.legend()
    plt.show()

    a = [(100, 100), (0, 200), (300, 300), (400, 400)]
    b = [(100, 100), (198, 200), (300, 295), (400, 400)]
    where_default = [i for i in range(17, 118, 25)]
    where_improved = [i for i in range(27, 128, 25)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in zip([0, 1], axes):
        ax.bar([20.0, 45.0, 70.0, 95.0], [j[i] for j in a], width=10, color='green', label="pysat")
        ax.bar([30.0, 55.0, 80.0, 105.0], [j[i] for j in b], width=10, color='orange', label="improved")
        ax.set(xticks=np.arange(25, 125, 25), yticks=np.arange(100, 450, 50))
        ax.set_xlabel('N')
        ax.set_ylabel('satisfied clauses')
        for j, val in enumerate(a):
            ax.text(where_default[j], 40 + 40 * j, str(val[i]), color="black", fontweight="bold")
        for j, val in enumerate(b):
            ax.text(where_improved[j], 40 + 40 * j, str(val[i]), color="white", fontweight="bold")
        ax.legend()
    plt.show()
