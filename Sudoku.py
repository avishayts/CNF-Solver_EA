from pprint import pprint


def v(i, j, d, n):
    return pow(n, 4) * (i - 1) + n * n * (j - 1) + d


def base_clauses(n):
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


def create_CNF(n, board):
    pprint(board)
    # solve a Sudoku problem
    clauses = base_clauses(n)
    for i in range(1, n*n+1):
        for j in range(1, n*n+1):
            d = board[i - 1][j - 1]
            # For each digit already known, a clause (with one literal).
            if d:
                clauses.append([v(i, j, d, n)])
    return clauses


def fill_board(n, board, assignment):
    for i in range(1, n*n+1):
        for j in range(1, n*n+1):
            for d in range(1, n*n+1):
                if assignment[v(i, j, d, n) - 1] == 1:
                    board[i - 1][j - 1] = d


board_9x9 = [[0, 0, 0, 1, 0, 9, 4, 2, 7],
             [1, 0, 9, 8, 0, 0, 0, 0, 6],
             [0, 0, 7, 0, 5, 0, 1, 0, 8],
             [0, 5, 6, 0, 0, 0, 0, 8, 2],
             [0, 0, 0, 0, 2, 0, 0, 0, 0],
             [9, 4, 0, 0, 0, 0, 6, 1, 0],
             [7, 0, 4, 0, 6, 0, 9, 0, 0],
             [6, 0, 0, 0, 0, 8, 2, 0, 5],
             [2, 9, 5, 3, 0, 1, 0, 0, 0]]

board_4x4 = [[2, 0, 0, 0],
             [0, 1, 0, 2],
             [0, 0, 3, 0],
             [0, 0, 0, 4]]

