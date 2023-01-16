import itertools
import random
import time
import math
from pprint import pprint

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorNFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from pysat.solvers import Solver
import Sudoku

global N
global M

global cnf

global POPULATION_SIZE
global ELITISM_RATE
global CROSSOVER_PROBABILITY
global MUTATION_PROBABILITY
global MUTATION_PROBABILITY_FOR_EACH
global TOURNAMENT_SIZE
global MAX_WORKERS


def gen_cnf(n):
    return [[random.randrange(1, n+1) if (random.random() > 0.5) else -random.randrange(1, 10) for _ in range(3)] for _ in range(M)]


def assignment_clause_count(assignment):
    count = 0
    for clause in cnf:
        for literal in clause:
            if (literal < 0 and assignment[abs(literal) - 1] == 0) or (literal > 0 and assignment[abs(literal) - 1] == 1):
                count += 1
                break
    return count


class CNFSolverEvaluator(SimpleIndividualEvaluator):
    def _evaluate_individual(self, individual):
        return assignment_clause_count(individual.vector)


def run():
    start = time.time()
    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=N),
                      population_size=POPULATION_SIZE,
                      # user-defined fitness evaluation method
                      evaluator=CNFSolverEvaluator(),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=ELITISM_RATE,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=CROSSOVER_PROBABILITY, k=1),
                          BitStringVectorNFlipMutation(probability=MUTATION_PROBABILITY,
                                                       probability_for_each=MUTATION_PROBABILITY_FOR_EACH, n=N)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=TOURNAMENT_SIZE, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=MAX_WORKERS,
        max_generation=1000,

        termination_checker=ThresholdFromTargetTerminationChecker(optimal=M, threshold=0.0)
        # statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()
    # Execute (show) the best solution
    return algo.execute()
    # return time.time() - start, algo.best_of_run_.get_pure_fitness()


def naive_solver():
    start = time.time()
    global_max = 0
    for seq in itertools.product([1, 0], repeat=N):
        local_max = assignment_clause_count(list(seq))
        global_max = max(global_max, local_max)
        if (global_max == M) or ((time.time() - start) > 10):
            break
    print(f"this is the max clauses which was satisfied by the naive algorithm: {global_max}\n"
          "!!note: if the max clauses was lessen then M then it might be due to limitation of time")
    return time.time() - start, global_max


def by_pysat():
    start = time.time()
    # get_cnf = CNF(from_clauses=cnf)
    # solver = Solver()
    solver = Solver(bootstrap_with=cnf)
    print('formula is', f'{"s" if solver.solve() else "uns"}atisfiable\n')
    # print('and the model is:', solver.get_model())
    return time.time() - start, solver.get_model()


def set_params(current_params):
    global POPULATION_SIZE
    global ELITISM_RATE
    global CROSSOVER_PROBABILITY
    global MUTATION_PROBABILITY
    global MUTATION_PROBABILITY_FOR_EACH
    global TOURNAMENT_SIZE
    global MAX_WORKERS
    POPULATION_SIZE = current_params[0]
    ELITISM_RATE = current_params[1]
    CROSSOVER_PROBABILITY = current_params[2]
    MUTATION_PROBABILITY = current_params[3]
    MUTATION_PROBABILITY_FOR_EACH = current_params[4]
    TOURNAMENT_SIZE = current_params[5]
    MAX_WORKERS = current_params[6]


def parameter_search(t_start):
    delta = 0.01*N
    min_time = t_start
    optimal_params = (POPULATION_SIZE, ELITISM_RATE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY,
                      MUTATION_PROBABILITY_FOR_EACH, TOURNAMENT_SIZE, MAX_WORKERS)
    params_ranges = [[i*2 for i in range(1, int(math.sqrt(N)))],
                     [i/N for i in range(1, N)],
                     [i/N for i in range(N)],
                     [i/N for i in range(N)],
                     [i/(2*N) for i in range(N)],
                     [i for i in range(1, 5)],
                     [i for i in range(1, 5)]]
    for i in range(N if N < 1000 else 1000):
        current_params = []
        for j in range(7):
            rand_index = random.randrange(len(params_ranges[j]))
            rand_param = params_ranges[j][rand_index]
            current_params.append(rand_param)
        set_params(current_params)
        current_run = run()
        current_time = current_run[0]
        current_fitness = current_run[1]
        if (current_time < min_time) and (M - current_fitness < delta):
            min_time = current_time
            optimal_params = tuple(current_params)
    return min_time, optimal_params


def conclusion(pysat_res_fitness, initial_fitness, res_fitness, naive_time, initial_time, res_of_params_search, pysat_time):
    print(f"runtime of naive solver (10 sec if interrupted): {10 if (naive_time > 10) else naive_time}\n"
          f"runtime of the initial parametric guess: {initial_time}\n"
          f"number of clauses which was satisfied: {initial_fitness}\n"
          f"runtime of the best parametric which was found: {res_of_params_search[0]}\n"
          f"number of clauses which was satisfied: {res_fitness}\n"
          f"the returned values (i.e the parameters which was found) were: {res_of_params_search[1]}\n"
          f"overall the runtime of the initial algorithm was {initial_time / res_of_params_search[0]} times longer\n"
          f"the runtime of the pysat_algorithm: {pysat_time}\n"
          f"number of clauses which was satisfied: {pysat_res_fitness}\n"
          f"the 'default' algorithm was {initial_time/pysat_time} times longer\n"
          f"and the 'improved' algorithm was {res_of_params_search[0]/pysat_time} times longer\n")


def collect_data():
    global M
    global N
    global cnf
    global POPULATION_SIZE
    global ELITISM_RATE
    global CROSSOVER_PROBABILITY
    global MUTATION_PROBABILITY
    global MUTATION_PROBABILITY_FOR_EACH
    global TOURNAMENT_SIZE
    global MAX_WORKERS

    experiment_loop = 10
    experiment_range = 5

    num_of_variables = [(i+1)*100 for i in range(experiment_range)]
    num_of_clauses = [(i+1)*200 for i in range(experiment_range)]

    default = [(0, 0) for _ in range(experiment_range)]
    improved = [(0, 0) for _ in range(experiment_range)]
    pysat = [(0, 0) for _ in range(experiment_range)]
    naive = [(0, 0) for _ in range(experiment_range)]

    params_improved = [(0, 0, 0, 0, 0, 0, 0) for _ in range(experiment_range)]

    for _ in range(experiment_loop):
        for i in range(experiment_range):
            N = num_of_variables[i]
            M = num_of_clauses[i]
            cnf = gen_cnf(N)

            POPULATION_SIZE = 20
            ELITISM_RATE = 1 / 300
            CROSSOVER_PROBABILITY = 0.5
            MUTATION_PROBABILITY = 0.05
            MUTATION_PROBABILITY_FOR_EACH = 0.03
            TOURNAMENT_SIZE = 3
            MAX_WORKERS = 6
            default_run = run()

            default_time = default_run[0]
            default_fitness = default_run[1]
            default[i] = (default[i][0] + default_time, default[i][1] + default_fitness)

            res = parameter_search(default_time)
            set_params(res[1])
            print(res[1])
            params_improved[i] = tuple(map(lambda i, j: i + j, res[1], params_improved[i]))

            improved_run = run()
            improved_time = improved_run[0]
            improved_fitness = improved_run[1]
            final_res = (improved_time, res[1])
            improved[i] = (improved[i][0] + improved_time, improved[i][1] + improved_fitness)

            pysat_res = by_pysat()
            pysat_runtime = pysat_res[0]
            pysat_assignment = [1 if (i > 0) else 0 for i in pysat_res[1]]
            pysat_fitness = assignment_clause_count(pysat_assignment)
            pysat[i] = (pysat[i][0] + pysat_runtime, pysat[i][1] + pysat_fitness)

            naive_res = naive_solver()
            naive_runtime = naive_res[0]
            naive_fitness = naive_res[1]
            naive[i] = (naive[i][0] + naive_runtime, naive[i][1] + naive_fitness)

            # conclusion(pysat_fitness, default_fitness, best_fitness, naive_runtime, default_time, final_res, pysat_runtime)

    default = [(collected_data[0]/experiment_loop, collected_data[1]/experiment_loop) for collected_data in default]
    improved = [(collected_data[0]/experiment_loop, collected_data[1]/experiment_loop) for collected_data in improved]
    pysat = [(collected_data[0]/experiment_loop, collected_data[1]/experiment_loop) for collected_data in pysat]
    naive = [(collected_data[0]/experiment_loop, collected_data[1]/experiment_loop) for collected_data in naive]

    params_improved = [tuple(map(lambda param: param/experiment_loop, collected_data)) for collected_data in params_improved]

    return default, improved, pysat, naive, params_improved


if __name__ == "__main__":
    # data = collect_data()
    # default_data = data[0]
    # improved_data = data[1]
    # pysat_data = data[2]
    # naive_data = data[3]
    # params_data = data[4]
    #
    # print(data[0])
    # print(data[1])
    # print(data[2])
    # print(data[3])
    # print(data[4])
    POPULATION_SIZE = 8
    ELITISM_RATE = 0.36
    CROSSOVER_PROBABILITY = 0.5784
    MUTATION_PROBABILITY = 0.4088
    MUTATION_PROBABILITY_FOR_EACH = 0.0607
    TOURNAMENT_SIZE = 4
    MAX_WORKERS = 4
    # n = 3
    # cnf = Sudoku.create_CNF(n, Sudoku.board_9x9)
    # N = pow(n*n, 3)
    # M = len(cnf)
    # run()
    n = 2
    cnf = Sudoku.create_CNF(n, Sudoku.board_4x4)
    N = pow(n*n, 3)
    M = len(cnf)
    print(M)
    assignment = run()
    Sudoku.fill_board(n, Sudoku.board_4x4, assignment)
    pprint(Sudoku.board_4x4)

