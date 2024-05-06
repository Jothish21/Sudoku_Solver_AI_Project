import numpy as np
from SudokuProblem import PROBLEM
from Solver import CSPSudokuSolver, SudokuSimulatedAnnealing

# creating an instance of CSPSudokuSolver
solver = CSPSudokuSolver(PROBLEM.copy())


# solving the sudoku puzzle using Constraint Propagation
if solver.solve_constraint_propagation():
    print("Sudoku puzzle solved using Constraint Propagation:")
    print(np.array(solver.puzzle).reshape((9, 9)))
else:
    print("No solution found for Sudoku puzzle using Constraint Propagation.")

# solving the sudoku puzzle using backtracking
if solver.solve_backtracking():
    print("Sudoku puzzle solved using Backtracking:")
    print(np.array(solver.puzzle).reshape((9, 9)))
else:
    print("No solution found for Sudoku puzzle using Backtracking.")

# solving the sudoku puzzle using backtracking with Degree Heuristic
if solver.solve_backtracking_with_degree_heuristic():
    print("Sudoku puzzle solved using Backtracking with Degree Heuristic:")
    print(np.array(solver.puzzle).reshape((9, 9)))
else:
    print("No solution found for Sudoku puzzle using Backtracking with Degree Heuristic.")


# creating an instance of SudokuSimulatedAnnealing
SAsolver = SudokuSimulatedAnnealing(PROBLEM.copy())

# solving the sudoku puzzle using Simulated Annealing
SAsolution = SAsolver.solve()
print("\n Sudoku puzzle solved using Simulated Annealing:")
print(np.array(SAsolution).reshape((9, 9)))
