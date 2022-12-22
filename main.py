import numpy as np
import pandas as pd

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


def CNF():
    return [[1, -2, 3], [2, 3, 4], [-1, -3, -4]]


def AssignmentClauseCount(assignment, cnf):
    count = 0
    for clause in cnf:
        for literal in clause:
            if (literal < 0 and assignment[abs(literal) - 1] == 0) or (
                    literal > 0 and assignment[abs(literal) - 1] == 1):
                count += 1
                break
    return count


class CNFSolverEvaluator(SimpleIndividualEvaluator):
    def _evaluate_individual(self, individual):
        """
            Compute the fitness value of a given individual.
            Parameters
            ----------
            individual: Vector
                The individual to compute the fitness value for.
            Returns
            -------
            float
                The evaluated fitness value of the given individual.
        """
        cnf = CNF()
        return AssignmentClauseCount(individual.vector, cnf)


def main():
    ass = [1, 1, 1, 0]
    print(AssignmentClauseCount(ass, CNF()))


if __name__ == "__main__":
    main()
