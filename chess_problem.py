import numpy as np
import random

def create_chessboard(genome, representation, size=8):

    chessboard = np.zeros((size, size), dtype=int)


    if representation == 'binary':

        binary_genome = np.concatenate(list(genome.genes.values()))
    
        if len(binary_genome.shape) == 1:
            binary_genome = binary_genome.reshape(size, size)
        
            chessboard[binary_genome==1] = 1

    elif representation == 'integer':

        integer_genome = next(iter(genome.genes.values()))

        for i, pos in enumerate(integer_genome):

                    row, col = np.unravel_index(pos, (size, size))
                    chessboard[row, col] = 1

    else:
        raise ValueError("Invalid representation")

    return chessboard




def fitness(individual,representation,size=8):
    chessboard = create_chessboard(individual,representation)
    total_pairs = size * (size - 1) // 2

    queens = np.array(np.where(chessboard == 1)).T
    board_of_collisions = np.zeros((size, size), dtype=int) #this board will mark the collisions, adding a point for each collision, so that if a queen has 2 collisions, it will have a 2 in the board_of_collisions

    # Iterate over each queen
    for queen in queens:

        y = queen[0]  # row number
        x = queen[1]  # column number

        # Check the row for collisions
        for i in range(size):
            if i != x and chessboard[y][i] == 1:  # Avoid checking with itself
                board_of_collisions[y][x] += 1
                
        # Check the column for collisions
        for i in range(size):
            if i != y and chessboard[i][x] == 1:  # Avoid checking with itself
                board_of_collisions[y][x] += 1

        # Check diagonals for collisions
        for i in range(size):
            for j in range(size):
                if i != y and j != x and abs(i - y) == abs(j - x) and chessboard[i][j] == 1:  # Avoid checking with itself and only check diagonal
                    board_of_collisions[y][x] += 1

    total_collisions = np.sum(board_of_collisions) / 2  # Each collision is counted twice so divide by 2
    fitness_score = total_pairs - total_collisions

    return fitness_score, board_of_collisions
