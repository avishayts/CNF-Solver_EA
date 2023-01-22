# CNF-Solver_EA
### Dor Zarka 316495357
### Avishay Tsur 316508431

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

## Solution Description
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
`assignment_clause_count(assignment)`: Recieves an assignment and count all the satisfiable clauses from the assignment.  
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

### Parameters search function  
`parameter_search(t_time, params_ranges, loop_num, indicator, son, dynamic_search)`:  

<ins>Non-Dynamic Search</ins>:  
This function tries to find the optimal parameters with the constants N and M. The searching is done in a random search, i.e., each iteration the function chooses randomly an optional value for each global parameter and runs the evolutionary algorithm with the random parameters.  
This was done intentionally in order to reduce significantly the runtime. Only if the random parameters achieved 'better' runtime and were no more less than 'delta' from the optimal fitness (i.e., M) they will be set globally.

<ins>Dynamic Search</ins>:  
if we assume that rather than the idea of a singular suitable set of global parameters, there are many ‘local’s maxima’  we can get  by different sets of global parameters, we may wish to keep on searching ‘locally’ after finding better new set of a global parameters. Nevertheless, we also wish to keep on searching for different ‘local’s maxima’.  
This idea is exactly what dynamic search is doing. If a new random set of parameters got better results than the last best set, dynamic search will search for loop_num/2 iterations on relatively small ranges around the new parameters which was found. If a new better local set  was found, then it will be set as the best new set and the function will search locally recursively.  After the local search ends, the search will keep on searching randomly on the full ranges.  

### Experiment functions
+ `collect_data()`: This function collects data on all the different runtime of the different algorithms (naïve algorithm, EC-KitY, EC-KitY after parameters search and pysat). This function generates a random CNF with N variables and M clauses at each iteration, and averages the sum of the runtimes obtained after 'experiment_loop' iterations. 
The function will measure the runtime for a range of different N (in a ratio of 1:2 with M). The range can be set by the variable 'experiment_range'.

+ `assignment_clause_count(assignment)`: Counts the number of the satisfied clauses.

+ `gen_cnf(n)`: Generates a random CNF clause.

+ `naive_solver()`: Solves the CNF clause with naive algorithm.

+ `by_pysat()`: Solves the CNF clause with pysat algorithm. 

+ `run()`: Run the evolutionary algorithm.  

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
+ **Selection:** `TournamentSelection(tournament_size=TOURNAMENT_SIZE, higher_is_better=True)`. Selection is done by tournament with TOURNAMENT_SIZE parameter that found in parameters search.

## Sudoku  
Sudoku problem is NP-Complete and has reduction to SAT problem.  
The reduction process is:  
1. Encoder: Takes Sudoku board with size $n^2 * n^2$ and generates CNF formula according the board.
2. CNF-Solver: Takes the generated CNF formula and finds satisfiabilty assigment to it.
3. Decoder: Takes the assignment from the CNF-Solver and fills the rest of the Sudoku board from it.

### Sudoku functions
+ `create_CNF(n, board)`: Gets a sudoku board with size $n^2 * n^2$, and generates from it CNF fromula.  

+ `num_of_variables(cnf)`: Gets a CNF formula and returns the number of different variables.  

+ `fill_board(n, board, assignment)`: Gets a sudoku board, and fills it cells according the assignment.  

+ `print_board(board)`: Gets a sudoku board and prints it.  

+ `v(i, j, d, n)`: Returns an integer according the cell and the value of the board. The calculation is: $pow(n, 4) * (i - 1) + n * n * (j - 1) + d$.  
The idea behind it is that for each cell $i,j$, we need to create $n*n$ variables.  

+ `map_to_index(literal)`: gets a literal and returns its corresponding index of the individual.  

## Sudoku run examples
## size $n = 2, 4*4$ board
<img width="345" alt="4x4_start" src="https://user-images.githubusercontent.com/77344388/213874934-ddd20b78-f19e-4936-8538-bb3397112d96.PNG">  

**Reduction Encoder**: The CNF formula that generated from the board is:  
$[[-19, -20], [-19, -14], [-17, -20], [-17, -18], [-16, -18], [-15, -19], [-15, -16], [-12, -17], [-12, -13], [-11, -13],$
$[-10, -15], [-10, -14], [-10, -11], [-8, -11], [-8, -9], [-8, -2], [-7, -16], [-7, -8], [-7, -1], [-5, -14],$
$[-5, -6], [-4, -9], [-3, -19], [-3, -5], [-3, -4], [-2, -13], [-2, -4], [-1, -18],[-1, -6], [-1, -2],$
$[1, 2], [3, 4], [5, 6], [7, 8], [9], [10, 11], [12, 13], [14], [15, 16], [17, 18], [19, 20]]$

This formula has **20 variables** and **41 clauses**.

<ins>Parameters:</ins>  
<img width="227" alt="Capture" src="https://user-images.githubusercontent.com/77344388/213907557-00e9376c-0017-4651-bab0-71bd4ec157c7.PNG">  

<ins>Graph of #Generations/#Unsasifiable clauses:</ins>  

![image](https://user-images.githubusercontent.com/77344388/213907636-d666e0cd-569f-46ac-ab68-6c6214a1e1a8.png)

**Result:** Before the evolutionary algroithm started to run, the number of unsatifiable clauses of the best individual's assigment (from the initial population) was 4.  
After 7 generations, there was an individual in the population that its assignment satisfied the CNF formula.

<ins>Board result:</ins>  
<img width="343" alt="1" src="https://user-images.githubusercontent.com/77344388/213907663-953078bb-c21d-4e27-b8a0-2838972e48cd.PNG">


## size $n = 3, 9*9$ board
<img width="366" alt="9x9_start" src="https://user-images.githubusercontent.com/77344388/213909414-0e839395-8a13-432b-8e0b-ec4db20f5259.PNG">

**Reduction Encoder**: The CNF formula that generated from the board is:  
$[[-85, -86], [-85, -82], [-84, -85], [-83, -87], [-83, -84], [-80, -84], [-80, -82], [-80, -81], [-79, -83], [-79, -81],$
$[-79, -80], [-78, -84], [-78, -82], [-78, -80], [-77, -83], [-77, -79], [-77, -78], [-74, -75], [-73, -75], [-70, -72],$
$[-70, -71], [-68, -76], [-68, -74], [-68, -69], [-67, -75], [-65, -84], [-65, -80], [-65, -66], [-65, -55], [-65, -43],$
$[-64, -67], [-64, -66], [-64, -65], [-64, -53], [-64, -41], [-63, -78], [-63, -65], [-63, -55], [-63, -43], [-63, -40],$
$[-62, -66], [-61, -62], [-59, -75], [-59, -67], [-59, -60], [-58, -85], [-57, -58], [-56, -67], [-56, -59], [-56, -58],$
$[-56, -57], [-55, -58], [-54, -55], [-53, -59], [-53, -56], [-53, -55], [-53, -54], [-52, -78], [-52, -65], [-52, -63],$
$[-52, -58], [-52, -55], [-52, -43], [-52, -40], [-51, -54], [-51, -52], [-50, -69], [-50, -62], [-49, -58], [-49, -55],$
$[-49, -52], [-49, -50], [-48, -59], [-48, -56], [-48, -53], [-48, -50], [-48, -49], [-47, -62], [-47, -50], [-46, -59],$
$[-46, -56], [-46, -53], [-46, -48], [-46, -47], [-45, -85], [-45, -58], [-44, -67], [-44, -59], [-44, -56], [-44, -45],$
$[-43, -55], [-43, -45], [-42, -43], [-41, -53], [-41, -44], [-41, -43], [-41, -42], [-40, -84], [-40, -80], [-40, -65],$
$[-40, -55], [-40, -45], [-40, -43], [-39, -83], [-39, -79], [-39, -42], [-39, -40], [-38, -64], [-38, -53], [-38, -44],$
$[-38, -41], [-38, -40], [-38, -39], [-37, -40], [-37, -39], [-37, -38], [-36, -48], [-36, -46], [-36, -44], [-36, -41],$
$[-36, -38], [-34, -73], [-34, -35], [-33, -54], [-32, -42], [-32, -33], [-31, -53], [-31, -41], [-31, -34], [-31, -33],$
$[-31, -32], [-30, -72], [-30, -33], [-30, -32], [-30, -31], [-29, -51], [-29, -33], [-28, -77], [-28, -32], [-28, -29],$
$[-28, -16], [-28, -12], [-27, -70], [-27, -30], [-27, -29], [-27, -28], [-27, -14], [-26, -33], [-26, -29], [-25, -48],$
$[-25, -34], [-25, -31], [-25, -26], [-24, -30], [-24, -27], [-24, -26], [-24, -25], [-23, -32], [-23, -28], [-22, -46],$
$[-22, -36], [-22, -34], [-22, -31], [-22, -25], [-22, -23], [-22, -10], [-22, -4], [-20, -73], [-20, -34], [-20, -21],$
$[-19, -57], [-19, -21], [-18, -56], [-18, -44], [-18, -34], [-18, -20], [-18, -19], [-17, -55], [-17, -43], [-16, -42],$
$[-16, -32], [-16, -17], [-15, -53], [-15, -41], [-15, -31], [-15, -20], [-15, -18], [-15, -17], [-15, -16], [-14, -72],$
$[-14, -30], [-14, -17], [-14, -16], [-14, -15], [-13, -84], [-13, -80], [-13, -65], [-13, -40], [-13, -17], [-12, -83],$
$[-12, -79], [-12, -39], [-12, -32], [-12, -16], [-12, -13], [-11, -64], [-11, -38], [-11, -31], [-11, -20], [-11, -18],$
$[-11, -15], [-11, -13], [-11, -12], [-10, -48], [-10, -25], [-10, -20], [-10, -18], [-10, -15], [-10, -11], [-9, -24],$
$[-9, -14], [-9, -10], [-8, -64], [-8, -38], [-8, -31], [-8, -15], [-8, -11], [-7, -62], [-6, -69], [-6, -50], [-6, -7],$
$[-5, -26], [-5, -6], [-4, -48], [-4, -25],[-4, -10], [-4, -8], [-4, -6], [-4, -5], [-3, -47], [-3, -7], [-3, -6],$
$[-2, -3], [-1, -46], [-1, -36], [-1, -25], [-1, -22], [-1, -10], [-1, -8], [-1, -4], [-1, -3], [-1, -2], [1, 2, 3],$
$[4, 5, 6], [7], [8], [9, 10], [11, 12, 13], [14, 15, 16, 17], [18, 19], [20, 21], [22, 23], [24, 25, 26], [27, 28, 29],$
$[30, 31, 32, 33], [34, 35], [36], [37, 38, 39, 40], [41, 42, 43], [44, 45], [46, 47], [48, 49, 50], [51, 52], [53, 54, 55],$
$[56, 57, 58], [59, 60], [61, 62], [63], [64, 65, 66], [67], [68, 69], [70, 71], [72], [73], [74, 75], [76], [77, 78],$
$[79, 80, 81], [82], [83, 84], [85, 86], [87]]$

This formula has **87 variables** and **292 clauses**.

<ins>Parameters:</ins>  
 <img width="319" alt="2121" src="https://user-images.githubusercontent.com/77344388/213909189-fc069937-9f30-4f70-9137-c77593170d1a.PNG">

<ins>Graph of #Generations/#Unsasifiable clauses:</ins>  

![image](https://user-images.githubusercontent.com/77344388/213909200-01d3f4eb-7a15-40bb-8231-44fd6b45f8fc.png)

**Result:** Before the evolutionary algroithm started to run, the number of unsatifiable clauses of the best individual's assigment (from the initial population) was more than 50 (out of 292).  
After less than 550 generations, there was an individual in the population that its assignment satisfied the CNF formula.

<ins>Board result:</ins>  
<img width="366" alt="3333333" src="https://user-images.githubusercontent.com/77344388/213909423-09e8e920-1670-48be-afa2-d5f3d1e8ffb4.PNG">

## Video
[![CNF-Solver_EA – main py 22_01_2023 16_18_08](https://user-images.githubusercontent.com/77344388/213920570-f466788f-1988-4471-92e1-4b99019f462d.png)
](https://www.youtube.com/watch?v=Wm8UGhWSJzU)

## Summary and Conclusion

## References
+ https://github.com/EC-KitY/EC-KitY
+ https://api.eckity.org/eckity.html
+ http://www.evolutionarycomputation.org/slides/
+ https://www.lri.fr/~conchon/ENSPSaclay/project/A_SAT-based_Sudoku_solver.pdf
