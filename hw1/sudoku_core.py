import numpy as np
import sys
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
    # print(sudoku)
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
    return None;

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
