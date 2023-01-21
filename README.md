# CNF-Solver_EA

## Introduction
One of the major unsolved issue of computer science is **P versus NP**.  
This issue asks if every problem whose solution can be quickly (aka: polynomial time) verified, can also be quickly solved.

<ins>Definitions:</ins>  
**P** contains all decision problems that can be solved by a deterministic Turing Machine using a polynomial time.  
**NP** contains all decision problems that can be solved by a non-deterministic Turing Machine using a polynomial time.  
**NP-Complete** contains all decision problems in NP that can be simulate by every other problem in NP. It other words, problem L is NP-Complete if L is NP, and from every problem L' that is NP there is a polynomial reduction from L' to L.  

![image](https://user-images.githubusercontent.com/77344388/212975790-0a016e88-d309-4975-8998-aa7eaa5f87fe.png)  

In 1971, *Stephen Cook and Leonid Levin* showed that if there is a NP-Complete problem that can be solved in polynomial time, then every NP problem can be solved in polynomial time also. It means that P = NP.  
Some of the well-known NP-Complete problems are:
+ Knapsack problem  
+ Hamiltonian path problem  
+ Travelling salesman problem (decision version)  
+ Subgraph isomorphism problem  
+ Subset sum problem  
+ Clique problem  
+ Vertex cover problem  
+ Independent set problem  
+ Dominating set problem  
+ Graph coloring problem  


One of the famous NP-Complete problems is **Boolean Satisfiability Problem**.  
Boolean Satistiability Problem (SAT) is the problem of determining if there exists an interpretation that satisfies a given Boolean formula.  
In other words, it asks whether the variables of a given Boolean formula can be consistently replaced by the values TRUE or FALSE in such way that the formula evaluates to TRUE. If this is the case, the formula is *satisfiable*, otherwise the formula is *unsatisfiable*.  

Every SAT formula can be converted to **Conjunctive Normal Form**.  
<ins>Definitions:</ins>  
**Literal** is an atomic formula. It is a variable or its negation. Examples: $x_1, \lnot x_4$.  
**Clause** is a disjunction of literals. Example: $x_2\vee v_3\vee\lnot v_4$.  
**Conjunctive Normal Form (CNF)** is a conjunction of clauses. Example: $(x_1\vee\lnot x_3\vee x_4\ )\land(\lnot x_1\vee x_2)$.  

## CNF-Solver_EA
CNF-Solver_EA is an evolutionary algorithm which get a CNF formula and finds satisfiable assignment.  

### Representation  
CNF formula represents as list of sub-lists of integers, such that each sub-list represents a clause, and each integer represents literal.  
The integer number represents the index of the variable. For example: 2 represent $x_2$.  
If the integer if positive, it represents the positive form of the variable, and if the integer is negative it represents and negative form of the variable. For example: 4 represents $x_4$, and -4 represents $\lnot x_4$.  
Entire CNF formula example:  
$(x_1\vee\lnot x_3\vee x_4\ )\land(\lnot x_1\vee x_2)$ will be represent as: $[[1,-3,4],[-1,2]]$.  

Each individual will be represented as bit string vector, such that the length of the vector is the count of the variables.  
For example: in $(x_1\vee\lnot x_3\vee x_4\ )\land(\lnot x_1\vee x_2)$ there are 4 variables, therefore the  bit string vector will be with size 4 and each index represents the corresponding variable. If the bit is 0 then the variable assigns False value, and if the bit is 1 then the variable assigns True value.  
For example: $[1,0,0,1]$ means that $x_1 = True, x_2 = False, x_3 = False, x_4 = True$.  

The population consists of M individuals, such that each individual is bit string vector as describes above.  

### Fitness Function
**assignment_clause_count(assignment)**: Recieves an assignment and count all the satisfiable clauses from the assignment.  
Higher fitness means higher satisfiable clauses.  
The maximum fitness is when all the clauses satistiable, means that the CNF formula is satisfiable.

### The evolution process
TODO

### Parameters search  
Given a CNF formula, we would like to find satisfiable assignment with the shortest time. Therefore, we need to find optimal parameters for the evolutionary algorithm.  
The parameters that we would like to get their optimal values for the algorithm are:
+ Population size
+ Elitism rate
+ Crossover probability for individual
+ Mutation probability for individual
+ Mutation probability for each bit in individual
+ Tournament size

#### Parameters search process  
TODO


## Sudoku  
Sudoku problem is NP-Complete and has reduction to SAT problem.  
The reduction process is:  
1. Encoder: Takes Sudoku board with size $n^2 * n^2$ and generates CNF formula according the board.
2. CNF-Solver: Takes the generated CNF formula and finds satisfiabilty assigment to it.
3. Decoder: Takes the assignment from the CNF-Solver and fills the rest of the Sudoku board from it.

### Sudoku functions
**create_CNF(n, board)**: Gets a sudoku board with size $n^2 * n^2$, and generates from it CNF fromula.  
**num_of_variables(cnf)**: Gets a CNF formula and returns the number of different variables.  
**fill_board(n, board, assignment)**: Gets a sudoku board, and fills it cells according the assignment.  
**print_board(board)**: Gets a sudoku board and prints it.  
**v(i, j, d, n)**: Returns an integer according the cell and the value of the board. The calculation is: $pow(n, 4) * (i - 1) + n * n * (j - 1) + d$.  
The idea behind it is that for each cell $i,j$, we need to create $n*n$ variables.  
**map_to_index(literal)**: gets a literal and returns its corresponding index of the individual.  

### Sudoku run example
