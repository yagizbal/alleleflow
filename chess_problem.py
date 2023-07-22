import numpy as np
from genome import *


def create_chessboard(genome, representation, size=8):

    chessboard = np.zeros((size+2, size+2), dtype=int)
    
    if representation == 'binary':
        
        # Map binary genome to expanded chessboard
        binary_genome = np.concatenate(list(genome.genes.values())).reshape(size, size)        
        chessboard[1:-1,1:-1] = binary_genome 
    
    elif representation == 'integer':
        
        # Map integer genome to expanded chessboard
        integer_genome = np.concatenate(list(genome.genes.values()))
        for i, pos in enumerate(integer_genome):
            row, col = np.unravel_index(pos, (size, size))
            chessboard[row+1, col+1] = 1

    return chessboard

def fitness_non_attacking(individual, representation, size=8):

    chessboard = create_chessboard(individual, representation, size)
    
    queens = np.array(np.where(chessboard == 1)).T
    
    total_pairs = size * (size - 1) // 2
    non_attacking_pairs = 0
    
    for i in range(len(queens)):
        for j in range(i+1, len(queens)):
            qi = queens[i]
            qj = queens[j]
            
            if (qi[0] != qj[0] and 
                qi[1] != qj[1] and
                abs(qi[0] - qj[0]) != abs(qi[1] - qj[1])):
                
                non_attacking_pairs += 1
                
    fitness = non_attacking_pairs
    
    return (fitness,)

def diagonal_attacks(chessboard, y, x):
    
    attacks = 0
    
    # Top right
    iy, ix = y-1, x+1
    while chessboard[iy, ix] == 1:
        attacks += 1
        iy -= 1
        ix += 1
        
    # Top left        
    iy, ix = y-1, x-1
    while chessboard[iy, ix] == 1:
        attacks += 1
        iy -= 1
        ix -= 1 
        
    # Bottom right
    iy, ix = y+1, x+1
    while chessboard[iy, ix] == 1:
        attacks += 1
        iy += 1
        ix += 1

    # Bottom left
    iy, ix = y+1, x-1
    while chessboard[iy, ix] == 1:
        attacks += 1
        iy += 1
        ix -= 1

    return attacks


test_population = Population(population_size=20, 
                             num_chromosomes=1,
                             chromosome_size=8, 
                             gene_type='integer', 
                             gene_range=(0,63),
                             no_overlap=True)

test_population.train(fitness_function=fitness_non_attacking, 
                      num_generations = 100000,
                      selection_pressure=0.04,
                      mutation_rate=0.05,
                        mutation_strength=0.5,
                        crossover_rate=0.5,
                        crossover_type='two_point',
                        replacement_rate=0.5,
                        verbose=True)
print(test_population.population[0].genes)

chessboard = create_chessboard(test_population.population[0], 'integer', size=8)
chessboard = chessboard[1:-1,1:-1]
print(chessboard)