import numpy as np
import sys
from pysat.formula import CNF
from pysat.solvers import MinisatGH
###
### Propagation function to be used in the recursive sudoku solver
###
def eliminate_rows(sudoku):
    solution = []
    for row in sudoku:
        certain = set([item[0] for item in row if len(item)==1])
        new_row = []
        for item in row:
            if len(item) ==1:
                new_row.append(item)
            else:
                possible = list(set(item)- certain)
                new_row.append(possible)
        solution.append(new_row)
    return solution

def eliminate_kxk(sudoku, k):
    steps = range(len(sudoku) // k)
    for i in steps:
        for j in steps:

            rows = sudoku[i*k:(i+1)*k]
            part = [item[j*k:(j+1)*k] for item in rows]
            part = [item for sublist in part for item in sublist]

            cleaned_part = eliminate_rows([part])[0]

            count = 0
            for l in range(i*k, (i+1)*k):
                for m in range(j*k, (j+1)*k):
                    sudoku[l][m] = sorted(cleaned_part[count])
                    count += 1

    return sudoku

def propagate(sudoku_possible_values,k):
    old = []
    reduced = sudoku_possible_values

    while not (old == reduced):
        old = reduced
        # rows
        reduced = eliminate_rows(reduced)

        # columns
        reduced = list(map(list, zip(*reduced))) # transpose
        reduced = eliminate_rows(reduced) # eliminate
        reduced = list(map(list, zip(*reduced))) # transpose back

        # 3x3
        reduced = eliminate_kxk(reduced, k)
        
        certains = 0
        for row in reduced:
            for item in row:
                if len(item) == 1:
                    certains += 1
        
    return reduced

###
### Solver that uses SAT encoding
###
def solve_sudoku_SAT(sudoku,k):
    k2 = k**2
    # certain_props = len([item for row in sudoku for item in row if item != 0])
    N_propositionals = k2**3
    print("number of propositions: ", N_propositionals)
    propositions = list(range(1, N_propositionals+1))

    propositions = np.array(propositions).reshape((k2, k2, k2))
    rules = CNF()

    ## Axiom trues
    for i, row in enumerate(sudoku):
        for j, item in enumerate(row):
            props = list(propositions[i,j])
            if item > 0:
                rules.append([int(props[item-1])])

     ## first rule each entry has a value, if it is set, that value must be True
    for i, row in enumerate(sudoku):
        for j, item in enumerate(row):
            rules.append([int(item) for item in propositions[i,j, :]])

    ## second rule each row value appears once
    for y in range(k2):
        for z in range(k2):
            for x in range(k2-1):
                for i in range(x+1, k2):
                    props = [propositions[x, y, z], propositions[i, y, z]]
                    rules.append([-int(item) for item in props])

    ## third rule each column value appears once
    for x in range(k2):
        for z in range(k2):
            for y in range(k2-1):
                for i in range(y+1, k2):
                    props = [propositions[x, y, z], propositions[x, i, z]]
                    rules.append([-int(item) for item in props])

    ## fourth rule each 3x3 value appears once
    for z in range(k2): # for all values
        for i in range(k): # x block
            for j in range(k): # y block
                for x in range(k): # x place in block
                    for y in range(k): # y place in block
                        for l in range(y+1, k): # for each
                            props = [propositions[k*i+x, k*j+y, z], propositions[k*i+x, k*j+l, z]]
                            rules.append([-int(item) for item in props])

                        for l in range(x+1, k):
                            for m in range(0, k):
                                props = [propositions[k*i+x, k*j+y, z], propositions[k*i+l, k*j+m, z]]
                                rules.append([-int(item) for item in props])

    ## There is at most one number in each entry
    for y in range(k2):
        for x in range(k2):
            for z in range(k2-1):
                for i in range(z+1, k2):
                    props = [propositions[x, y, z], propositions[x, y, i]]
                    rules.append([-int(item) for item in props])
    
    ## each number appears at least once in row and column and 3x3
    for y in range(k2):
        for z in range(k2):
            rules.append([int(item) for item in propositions[:, y, z]])
            
    for x in range(k2):
        for z in range(k2):
            rules.append([int(item) for item in propositions[x, :, z]])

    for i in range(k): # x block
        for j in range(k): # y block
            for x in range(k): # x place in block
                for y in range(k): # y place in block
                    rules.append([int(item) for item in propositions[k*i + x, k*j+y, :]])
    # print("Number of rules in total", len(rules))
    # formulas = CNF()
    # [formulas.append(rule) for rule in rules]

    solver = MinisatGH();
    solver.append_formula(rules);
    answer = solver.solve();
    if answer == True:
        for i, lit in enumerate(solver.get_model()):
            if lit > 0:
                idx = lit -1
                sudoku[(idx // k2) // k2][(idx // k2)%k2] = (idx % k2)+1 
                # print((idx //9) // 9, (idx // 9)%9, (idx % 9)+1)
                # print(type(lit))
                # print("Variable {} is true".format(lit));
                
    else:
        print("Did not find a model!");

    return sudoku
###
### Solver that uses CSP encoding
###
def solve_sudoku_CSP(sudoku,k):
    return None;

###
### Solver that uses ASP encoding
###
def solve_sudoku_ASP(sudoku,k):
    return None;

###
### Solver that uses ILP encoding
###
def solve_sudoku_ILP(sudoku,k):
    return None;
