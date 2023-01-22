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

## Problem Description
The problem is to find satisfiable assignment to a given CNF formula as described above.  
The naive solution is to run over all the $2^n$ possibilities of the values of the variables, when $n$ is the amount of the variables.  
This approach is very not efficient and when $n$ is very large, it might run for a very long time.  
In this project we will show a solution to solve CNF formula by Evolutionary Algorithm.  

## Problem Solution
We will solve the CNF formula and find a satisfiable assignment by Evolutionary Algorithm

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

## Parameters search  
Given a CNF formula, we would like to find satisfiable assignment with the shortest time. Therefore, we need to find optimal parameters for the evolutionary algorithm.  
The parameters that we would like to get their optimal values for the algorithm are:
+ Population size
+ Elitism rate
+ Crossover probability for individual
+ Mutation probability for individual
+ Mutation probability for each bit in individual
+ Tournament size

### Parameters search functions  
+ **parameter_search(t_time):** This function gets an initial time which was received say from some initial run, and it try to find the optimal parameters with the constants N and M. the searching is done in a random search, i.e. each iteration the function choose randomly an optional value for each global parameter and runs the evolutionary algorithm with the random parameters. This was done intentionally in order to reduce significantly the runtime. Only if the random parameters achieved 'better' runtime and were no more less than 'delta' from the optimal fitness (i.e. M) they will be set globally.

+ **collect_data():** This function collects data on all the different runtime of the different algorithms (naïve algorithm, EC-KitY, EC-KitY after parameters search and pysat). This function generates a random CNF with N variables and M clauses at each iteration, and averages the sum of the runtimes obtained after 'experiment_loop' iterations. 
The function will measure the runtime for a range of different N (in a ratio of 1:2 with M). The range can be set by the variable 'experiment_range'.

+ **assignment_clause_count(assignment):** Counts the number of the satisfied clauses.

+ **gen_cnf(n):** Generates a random CNF clause.

+ **naive_solver():** Solves the CNF clause with naive algorithm.

+ **by_pysat ():** Solves the CNF clause with pysat algorithm. 

+ **run():** Run the evolutionary algorithm.  

**Comparison between different algorithms and different parameters:**  
![image](https://user-images.githubusercontent.com/77344388/213907186-ca9b4af6-54db-4fb1-a962-3c62aa2449ab.png)
![image](https://user-images.githubusercontent.com/77344388/213907187-9ec49f65-38e4-4418-ae1e-3ca811c2d0a0.png)
![image](https://user-images.githubusercontent.com/77344388/213907192-4d13a960-5422-4bcb-b451-fe52c0524e07.png)
![image](https://user-images.githubusercontent.com/77344388/213907193-17c82208-08b6-45bf-baab-a88cc3333e53.png)
![image](https://user-images.githubusercontent.com/77344388/213907194-6174d952-702d-4811-a4a7-810527346562.png)
![image](https://user-images.githubusercontent.com/77344388/213907198-c7de6236-17c0-4d74-a72b-f9dc0d54051a.png)
![image](https://user-images.githubusercontent.com/77344388/213907199-9675fe72-db95-4367-a695-7b100a5b2069.png)
![image](https://user-images.githubusercontent.com/77344388/213907202-dd58cf7c-b3de-46ff-9b87-9f44dc08d2c0.png)
![image](https://user-images.githubusercontent.com/77344388/213907203-d46ac930-c7e1-4b6f-a417-6e4431cbbb24.png)
![image](https://user-images.githubusercontent.com/77344388/213907207-bf3f53b8-fab6-4f29-aae9-d4b07bd6056a.png)

**Conclusion**
+ We can see that the runtime of the impoved run (with the new parameters) decreased in comparison to the default parameters.  
+ The naive algorithm's runtime is very high.
+ There are algorithms that are very efficient such as pysat.  
+ Then we want to solve a CNF formula, we can search for parameters once, and then use those parameters for further runs when the CNF-formula is with the same size order.

## Software Overview
In order to use the CNF-Solver_EA to solve NP-Complete problem, the user must define the reduction from the problem to CNF formula.  
It means that the user need to define encoder and decoder from the problem.  
Then the user might want to search for optimal parameters as described above. We highly recommend to use this because without using optimization it might not solve the CNF formula, and if it does solve, we want it to be fast as it can be.

The CNF-Solver uses Evolutionary Algorithm by EC-KitY library.  
The algorithm uses:  
+ **Population and Representation:** `GABitStringVectorCreator(length=N)`. Each individual is bit string vector with size N (number of variables in formula).  
+ **Evaluator and Fitness:** `CNFSolverEvaluator()`. Returns the count of the satisfiable clauses by the individual's assignment vector. Higher is better.  
+ **Cross-over:** `VectorKPointsCrossover(probability=CROSSOVER_PROBABILITY, k=1)`. 1-point cross-over, with the optimal probability that found in parameters search.  
+ **Mutation:** `BitStringVectorNFlipMutation(probability=MUTATION_PROBABILITY, probability_for_each=MUTATION_PROBABILITY_FOR_EACH, n=N)`. For each individual it makes mutation with probability MUTATION_PROBABILITY, and flips each bit with probability MUTATION_PROBABILITY_FOR_EACH. The parameters are the ones that found in parameters search.

## Sudoku  
Sudoku problem is NP-Complete and has reduction to SAT problem.  
The reduction process is:  
1. Encoder: Takes Sudoku board with size $n^2 * n^2$ and generates CNF formula according the board.
2. CNF-Solver: Takes the generated CNF formula and finds satisfiabilty assigment to it.
3. Decoder: Takes the assignment from the CNF-Solver and fills the rest of the Sudoku board from it.

### Sudoku functions
+ **create_CNF(n, board)**: Gets a sudoku board with size $n^2 * n^2$, and generates from it CNF fromula.  

+ **num_of_variables(cnf)**: Gets a CNF formula and returns the number of different variables.  

+ **fill_board(n, board, assignment)**: Gets a sudoku board, and fills it cells according the assignment.  

+ **print_board(board)**: Gets a sudoku board and prints it.  

+ **v(i, j, d, n)**: Returns an integer according the cell and the value of the board. The calculation is: $pow(n, 4) * (i - 1) + n * n * (j - 1) + d$.  
The idea behind it is that for each cell $i,j$, we need to create $n*n$ variables.  

+ **map_to_index(literal)**: gets a literal and returns its corresponding index of the individual.  

## Sudoku run example
### size $n = 2, 4*4$ board
<img width="345" alt="4x4_start" src="https://user-images.githubusercontent.com/77344388/213874934-ddd20b78-f19e-4936-8538-bb3397112d96.PNG">  
<ins>Parameters:</ins>  
<img width="227" alt="Capture" src="https://user-images.githubusercontent.com/77344388/213907557-00e9376c-0017-4651-bab0-71bd4ec157c7.PNG">  
<ins>Graph of #Generations/#Unsasifiable clauses:</ins>  

![image](https://user-images.githubusercontent.com/77344388/213907636-d666e0cd-569f-46ac-ab68-6c6214a1e1a8.png)

<ins>Board result:</ins>  
<img width="343" alt="1" src="https://user-images.githubusercontent.com/77344388/213907663-953078bb-c21d-4e27-b8a0-2838972e48cd.PNG">


### size $n = 3, 9*9$ board
