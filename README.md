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

One of the famous NP-Complete problems is **Boolean Satisfiability Problem**.  
Boolean Satistiability Problem (SAT) is the problem of determining if there exists an interpretation that satisfies a given Boolean formula.  
In other words, it asks whether the variables of a given Boolean formula can be consistently replaced by the values TRUE or FALSE in such way that the formula evaluates to TRUE. If this is the case, the formula is *satisfiable*, otherwise the formula is *unsatisfiable*.  

Every SAT formula can be converted to **Conjunctive Normal Form**.  
<ins>Definitions:</ins>  
**Literal** is an atomic formula. It is a variable or its negation. Examples: $x_1, \lnot x_4$.  
**Clause** is a disjunction of literals. Example: $x_2\vee v_3\vee\lnot v_4$.  
**Conjunctive Normal Form (CNF)** is a conjunction of clauses. Example: $(x_1\vee\lnot x_3\vee x_4\ )\land(\lnot x_1\vee x_2)$.  
