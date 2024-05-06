import numpy as np
import random
from simanneal import Annealer

# creating a class for CSP algorithms
class CSPSudokuSolver:
    # initializing sudoku puzzle and number of assignments counter
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.num_assignments = 0

    # calling backtracking to solve puzzle
    def solve_backtracking(self):
        # resetting counter to 0
        self.num_assignments = 0
        if self.backtrack():
            return True # puzzle solved yay!
        else:
            return False # puzzle not solved nay!
        
    # calling constraint propagation to solve puzzle
    def solve_constraint_propagation(self):
        # resetting counter to 0
        self.num_assignments = 0
        if self.propagate_constraints():
            if self.backtrack():
                return True
        return False

    # calling backtracking with degree heuristic to solve puzzle
    def solve_backtracking_with_degree_heuristic(self):
        self.num_assignments = 0
        if self.backtrack_degree_heuristic():
            return True
        else:
            return False

    def backtrack(self):
        # finding an empty cell in puzzle
        empty_cell = self.find_empty()
        if not empty_cell:
            return True  # if found no empty cells left, then puzzle is solved
        
        row, col = empty_cell # extracting row and column indices of empty cell

        # looping through number 1 to 9
        for num in range(1, 10):
            # check if the current number is valid to to use in the empty cell as per sudoku rules
            if self.is_valid(row, col, num):
                # if valid then keep the number in that empty cell and increament counter
                self.puzzle[row * 9 + col] = num
                self.num_assignments += 1
                
                # recursively call backtracking to continue solving
                if self.backtrack():
                    return True  # found a solution

                self.puzzle[row * 9 + col] = 0  # if no solution found, then reset the cell to 0 (backtrack)
        return False  # return false if no solution found at all!!!(thats a problem now!!!)

    def propagate_constraints(self):
        # looping till no progress can be made
        while True:
            stalled = True # a flag to tell we can no longer progress further
            # iterating through each cell in puzzle
            for i in range(9):
                for j in range(9):
                    if self.puzzle[i * 9 + j] == 0: # checking if cell is empty
                        domain = self.get_domain(i, j) # getting domain for cell
                        # if domain is empty then the puzzle is inconsistent
                        if len(domain) == 0:
                            return False  
                        # assigning the value, is there is only 1 value in the domain
                        if len(domain) == 1:
                            self.puzzle[i * 9 + j] = domain[0]
                            self.num_assignments += 1
                            stalled = False # stalling as we progressed
            # exiting loop if no progress is made at all
            if stalled:
                return True  
            
    def get_domain(self, row, col):
        # initializing domain with all possible values(1 to 9)
        domain = list(range(1, 10))

        for i in range(9):
            # remove values that are already present in same row
            if self.puzzle[row * 9 + i] in domain:
                domain.remove(self.puzzle[row * 9 + i])
            # remove values that are already present in same column
            if self.puzzle[i * 9 + col] in domain:
                domain.remove(self.puzzle[i * 9 + col])

        # getting the starting row and column of the sub grid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)

        # remove values that are already in the sub grid
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.puzzle[i * 9 + j] in domain:
                    domain.remove(self.puzzle[i * 9 + j])
        
        return domain

    def backtrack_degree_heuristic(self):
        # find empty cells and prioritize them based on size of their domains
        empty_cells = [(i, j) for i in range(9) for j in range(9) if self.puzzle[i * 9 + j] == 0]
        if not empty_cells:
            return True  # Puzzle solved
        # choosing domain with smallest domain(degree heuristic)
        row, col = min(empty_cells, key=lambda x: len(self.get_domain(x[0], x[1])))
        domain = self.get_domain(row, col)

        # try values from the domain for choosen cell
        for value in domain:
            if self.is_valid(row, col, value):
                self.puzzle[row * 9 + col] = value # assinging value to cell and proceeding with back tracking
                self.num_assignments += 1
                if self.backtrack_degree_heuristic():
                    return True
                # if solution is not found, then backtract
                self.puzzle[row * 9 + col] = 0  
        return False  

    def count_empty_neighbors(self, row, col):
        count = 0
        # couting empty cells in the same row and column
        for i in range(9):
            if self.puzzle[row][i] == 0:
                count += 1
            if self.puzzle[i][col] == 0:
                count += 1
        
        # finding starting row and column of the sub grid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        # looping to count the empty cells in sub grid
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.puzzle[i][j] == 0:
                    count += 1



    # function to find an empty cell
    def find_empty(self):
        # iterating through each cell in the puzzle
        for i in range(9):
            for j in range(9):
                # if cell in empty, return its indices
                if self.puzzle[i * 9 + j] == 0:
                    return (i, j)  
        return None  # returning None if no empty cell is found

    def is_valid(self, row, col, num):
        # check if number is already in the same row or column
        for i in range(9):
            if self.puzzle[row * 9 + i] == num or self.puzzle[i * 9 + col] == num:
                return False
        
        # finding the starting indices of sub grid of the cell
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)

        # checking if the number is there in the same 3x3 box
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.puzzle[i * 9 + j] == num:
                    return False
        
        return True # if number assigned is valid as per sudoku rules then return True

    def get_num_assignments(self):
        return self.num_assignments
    

# creating a class for Simulated Annealing
class SudokuSimulatedAnnealing(Annealer):
    # initialing sudoku puzzle, size and assignment counter
    def __init__(self, problem):
        self.problem = problem
        self.size = 9 # size of sudoku puzzle
        self.num_assignments = 0 # number of assignments counter
        # calling the superclass constructor with initial solution
        super().__init__(initial_state=self.initial_solution())

    # making a initial solution to use as starting point to solve with SA
    def initial_solution(self):
        solution = self.problem.copy() # creating a copy of puzzle
        for block in range(9): # going through each 3x3 block
            indices = self.get_block_indices(block) # getting indices of cells in current block
            block_values = self.problem[indices] # extracting values of indices
            # finding indices of empty cells
            empty_indices = [i for i in indices if self.problem[i] == 0]
            num_to_fill = [i for i in range(1, 10) if i not in block_values] # generating numbers to fill in empty cells
            random.shuffle(num_to_fill) # randomly suffling to filling
            # filling empty cells with shuffled numbers
            for index, value in zip(empty_indices, num_to_fill):
                solution[index] = value
                self.num_assignments += 1
        return solution

    # getting the single flat index representing cells in the grid
    def get_row_col(self, row, col):
        return row * self.size + col

    # getting indices of all cells in a block
    def get_block_indices(self, block_num):
        firstrow = (block_num // 3) * 3 # calculating the 1st row index of block
        firstcol = (block_num % 3) * 3 # calculating the 1st column index of block
        indices = [] # empty list to store indices
        
        # iterating through each cell
        for i in range(3):
            for j in range(3):
                # calculating the flat index of cell using row and col indices
                index = self.get_row_col(firstrow + i, firstcol + 1)
                indices.append(index)
        return indices

    # performing move operation
    def move(self):
        # randomly selecting a block
        block = random.randrange(9)
        # getting indices of empty cells with blocks
        indices = [i for i in self.get_block_indices(block) if self.problem[i] == 0]
        m, n = random.sample(indices, 2) # randomly selecting 2 indices from list of empty cells
        # swapping values at selected indices
        self.state[m], self.state[n] = self.state[n], self.state[m]
        self.num_assignments += 1

    # calculating the energy(objective function) of current state
    def energy(self):
        # making a lambda funtion to calculate the score for each column and row
        column_score = lambda n: -len(set(self.state[self.get_row_col(i, n)] for i in range(self.size)))
        row_score = lambda n: -len(set(self.state[self.get_row_col(n, i)] for i in range(self.size)))

        # calculating total energy by summing scores for each column and row
        score = sum(column_score(n) + row_score(n) for n in range(self.size))
        return score

    def solve(self):
        # setting the parameters for simulated annealing
        self.Tmax = 20 # initial temperature
        self.Tmin = 5 # final temperature or cooling temp
        self.steps = 1000000 # number of steps 
        self.updates = 1000 # number of updates
        state, e = self.anneal() # performing SA and returing final state
        return state
    
    def get_num_assignments(self):
        return self.num_assignments