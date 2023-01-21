from pprint import pprint
import itertools


def v(i, j, d, n):
    return pow(n, 4) * (i - 1) + n * n * (j - 1) + d


def base_clauses(n, board):
    res = []
    # for all cells, ensure that the each cell:
    for i in range(1, n*n+1):
        for j in range(1, n*n+1):
            # denotes (at least) one of the 9 digits (1 clause)
            res.append([v(i, j, d, n) for d in range(1, n*n+1)])
            # does not denote two different digits at once (36 clauses)
            for d in range(1, n*n+1):
                for dp in range(d + 1, n*n+1):
                    res.append([-v(i, j, d, n), -v(i, j, dp, n)])

    def valid(cells):
        for i, xi in enumerate(cells):
            for j, xj in enumerate(cells):
                if i < j:
                    for d in range(1, n*n+1):
                        res.append([-v(xi[0], xi[1], d, n), -v(xj[0], xj[1], d, n)])

    # ensure rows and columns have distinct values
    for i in range(1, n*n+1):
        valid([(i, j) for j in range(1, n*n+1)])
        valid([(j, i) for j in range(1, n*n+1)])

    # ensure 3x3 sub-grids "regions" have distinct values
    for i in range(1, n*n+1, n):
        for j in range(1, n*n+1, n):
            valid([(i + k % n, j + k // n) for k in range(n*n)])

    # assert len(res) == 81 * (1 + 36) + 27 * 324
    return res


def remove_from_board(n, board, clauses):
    for i in range(1, n*n+1):
        for j in range(1, n*n+1):
            d = board[i - 1][j - 1]
            if d != 0:  # remove clauses and literals s.t built-in cells won't change
                val = v(i, j, d, n)
                for clause in clauses[:]:
                    if val in clause:
                        clauses.remove(clause)
                    else:
                        if -val in clause:
                            clause.remove(-val)
                        for dp in range(1, n*n+1):
                            val_dp = v(i, j, dp, n)
                            if val_dp in clause:
                                clause.remove(val_dp)
                            if -val_dp in clause:
                                clause.remove(-val_dp)
                        for r in range(1, n*n+1):
                            val_r = v(r, j, d, n)
                            if val_r in clause:
                                clause.remove(val_r)
                            if -val_r in clause:
                                clause.remove(-val_r)
                        for c in range(1, n*n+1):
                            val_c = v(i, c, d, n)
                            if val_c in clause:
                                clause.remove(val_c)
                            if -val_c in clause:
                                clause.remove(-val_c)
                        for r in range(1, n+1):
                            for c in range(1, n+1):
                                val_cr = v(int((i-1)/n)*n+r, int((j-1)/n)*n+c, d, n)
                                if val_cr in clause:
                                    clause.remove(val_cr)
                                if -val_cr in clause:
                                    clause.remove(-val_cr)
                        if len(clause) == 0:
                            clauses.remove(clause)
                        elif len(clause) == 1 and clause[0] < 0:
                            clauses.remove(clause)
    clauses.sort()
    return list(k for k, _ in itertools.groupby(clauses))


def num_of_variables(clauses):
    flat_clauses = [item for sublist in clauses for item in sublist]
    set_clauses = set(flat_clauses)
    # print(set_clauses)
    return int(len(set_clauses)/2)


def create_CNF(n, board):
    pprint(board)
    # solve a Sudoku problem
    clauses = base_clauses(n, board)
    # for i in range(1, n*n+1):
    #     for j in range(1, n*n+1):
    #         d = board[i - 1][j - 1]
    #         # For each digit already known, a clause (with one literal).
    #         if d:
    #             clauses.append([v(i, j, d, n)])
    #             for k in range (1, n*n+1):
    #                 if k != d:
    #                     clauses.append([-v(i, j, k, n)])
    # pprint(clauses)
    clauses = remove_from_board(n, board, clauses)
    # pprint(clauses)
    return clauses


def map_to_index(cnf, literal):
    flat_clauses = [item for sublist in cnf for item in sublist]
    set_clauses = list(set(flat_clauses))
    set_clauses = list(filter(lambda i: i >= 0, set_clauses))
    set_clauses.sort()
    return set_clauses.index(literal)


def fill_board(n, board, assignment, cnf):
    flat_clauses = [item for sublist in cnf for item in sublist]
    set_clauses = list(set(flat_clauses))
    set_clauses = list(filter(lambda i: i >= 0, set_clauses))
    set_clauses.sort()
    for i in range(1, n*n+1):
        for j in range(1, n*n+1):
            for d in range(1, n*n+1):
                if set_clauses.count(v(i, j, d, n)) != 0:
                    if assignment[set_clauses.index(v(i, j, d, n))] == 1:
                        board[i - 1][j - 1] = d


board_9x9 = [[0, 6, 0, 1, 0, 9, 4, 2, 7],
             [1, 0, 9, 8, 0, 7, 0, 0, 6],
             [0, 0, 7, 0, 5, 0, 1, 0, 8],
             [0, 5, 6, 9, 0, 0, 0, 8, 2],
             [0, 0, 0, 6, 2, 0, 0, 4, 0],
             [9, 4, 0, 0, 0, 5, 6, 1, 0],
             [7, 0, 4, 0, 6, 0, 9, 0, 0],
             [6, 0, 0, 0, 0, 8, 2, 0, 5],
             [2, 9, 5, 3, 0, 1, 0, 6, 0]]

board_4x4 = [[2, 0, 0, 0],
             [0, 1, 0, 2],
             [0, 0, 3, 0],
             [0, 0, 0, 4]]

