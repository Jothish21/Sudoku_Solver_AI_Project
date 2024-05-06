import time
from SudokuProblem import PROBLEM
from Solver import CSPSudokuSolver, SudokuSimulatedAnnealing

# defining functions to solve sudoku using constraint propagation, backtracking, backtracking with degree heuristic
# and Simulated annealing

# solving puzzle using Constrain Propagation
def call_constraint_propagation(problem):
    solver = CSPSudokuSolver(problem.copy()) # creating an instance of CSPSudokuSolver
    start_time = time.time() # execution start time
    if solver.solve_constraint_propagation(): # solving puzzle
        end_time = time.time() # execution end time
        exec_time = end_time - start_time # total execution time
        return solver.get_num_assignments(), exec_time # returing number of assignments and execution time
    else:
        return None

def call_backtracking(problem):
    solver = CSPSudokuSolver(problem.copy()) # creating an instance of CSPSudokuSolver
    start_time = time.time() # execution start time
    if solver.solve_backtracking(): # solving puzzle
        end_time = time.time() # execution end time
        exec_time = end_time - start_time # total execution time
        return solver.get_num_assignments(), exec_time # returing number of assignments and execution time
    else:
        return None

def call_backtracking_degree_heuristic(problem):
    solver = CSPSudokuSolver(problem.copy()) # creating an instance of CSPSudokuSolver
    start_time = time.time() # execution start time
    if solver.solve_backtracking_with_degree_heuristic(): # solving puzzle
        end_time = time.time() # execution end time
        exec_time = end_time - start_time # total execution time
        return solver.get_num_assignments(), exec_time # returing number of assignments and execution time
    else:
        return None

def call_simulated_annealing(problem):
    solver = SudokuSimulatedAnnealing(problem) # creating an instance of SudokuSimulatedAnnealing
    start_time = time.time() # execution start time
    solution = solver.solve() # solving puzzle
    end_time = time.time() # execution end time
    exec_time = end_time - start_time # total execution time
    num_assignments = solver.get_num_assignments() 
    return num_assignments, exec_time # returing number of assignments and execution time

# Printing out the number of assignments and execution times for all the algorithms

constraint_propagation_result = call_constraint_propagation(PROBLEM)
if constraint_propagation_result:
    num_assignments_constraint_propagation, exec_time_constraint_propagation = constraint_propagation_result
    print("Constraint Propagation Algorithm:")
    print("Number of Assignments:", num_assignments_constraint_propagation)
    print("Execution Time:", exec_time_constraint_propagation, "seconds\n")
else:
    print("No solution found for the Sudoku puzzle using Constraint Propagation algorithm.\n")

backtracking_result = call_backtracking(PROBLEM)
if backtracking_result:
    num_assignments_backtracking, exec_time_backtracking = backtracking_result
    print("Backtracking Algorithm:")
    print("Number of Assignments:", num_assignments_backtracking)
    print("Execution Time:", exec_time_backtracking, "seconds\n")
else:
    print("No solution found for the Sudoku puzzle using Backtracking algorithm.\n")
    
backtracking_DH_result = call_backtracking_degree_heuristic(PROBLEM)
if backtracking_DH_result:
    num_assignments_backtracking_DH, exec_time_backtracking_DH = backtracking_DH_result
    print("Backtracking Algorithm with Degree Heuristic:")
    print("Number of Assignments:", num_assignments_backtracking_DH)
    print("Execution Time:", exec_time_backtracking_DH, "seconds\n")
else:
    print("No solution found for the Sudoku puzzle using Backtracking algorithm with Degree Heuristic.\n")

sa_assignments, sa_exec_time = call_simulated_annealing(PROBLEM)
print("\n Simulated Annealing:")
print("Number of Assignments:", sa_assignments)
print("Execution Time:", sa_exec_time, "seconds\n")