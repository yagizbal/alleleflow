import numpy as np
from population import *

def create_chessboard(genome, representation, size=8):
    chessboard = np.zeros((size, size), dtype=int)
    
    if representation == 'binary':
        binary_genome = np.concatenate(genome.genes) # flatten to 1D array
        binary_genome = binary_genome.reshape(size, size) # reshape to 8x8
        for i in range(size):
            for j in range(size):
                if binary_genome[i,j] == 1:
                    chessboard[i, j] = 1
    elif representation == 'integer':
        for i, pos in enumerate(genome.genes):
            row, col = np.unravel_index(pos, (size, size))
            chessboard[row, col] = 1
    return chessboard

def diagonal_attacks(chessboard, y, x):
    attacks = 0
    size = chessboard.shape[0]
    # Top right
    iy, ix = y-1, x+1
    while iy >= 0 and ix < size:
        if chessboard[iy, ix] == 1:
            attacks += 1
            break
        iy -= 1
        ix += 1
    # Top left        
    iy, ix = y-1, x-1
    while iy >= 0 and ix >= 0:
        if chessboard[iy, ix] == 1:
            attacks += 1
            break
        iy -= 1
        ix -= 1 
    # Bottom right
    iy, ix = y+1, x+1
    while iy < size and ix < size:
        if chessboard[iy, ix] == 1:
            attacks += 1
            break
        iy += 1
        ix += 1
    # Bottom left
    iy, ix = y+1, x-1
    while iy < size and ix >= 0:
        if chessboard[iy, ix] == 1:
            attacks += 1
            break
        iy += 1
        ix -= 1
    return attacks

def fitness_non_attacking(individual, representation, size=8):
    chessboard = create_chessboard(individual, representation, size)
    queens = np.array(np.where(chessboard == 1)).T
    total_pairs = len(queens) * (len(queens) - 1) // 2
    attacking_pairs = 0
    for i in range(len(queens)):
        for j in range(i+1, len(queens)):
            qi = queens[i]
            qj = queens[j]
            if qi[0] == qj[0] or qi[1] == qj[1] or diagonal_attacks(chessboard, qi[0], qi[1]) > 0 or diagonal_attacks(chessboard, qj[0], qj[1]) > 0:
                attacking_pairs += 1
    for queen in queens:
        if diagonal_attacks(chessboard, queen[0], queen[1]) > 0:
            attacking_pairs += 1
    fitness = total_pairs - attacking_pairs
    return (fitness,)


"""
test_population = Population(population_size=100, 
                             num_chromosomes=1,
                             chromosome_size=8, 
                             representation='integer', 
                             gene_range=(0,63),
                             no_overlap=True)

fitness_history = test_population.train(fitness_function=fitness_non_attacking,
                      num_generations=3000,
                      selection_pressure=0.1,
                      mutation_rate=0.2,
                      mutation_type = "uniform",
                      mutation_strength=0.2,
                      crossover_rate=0.2,
                      crossover_type='one_point',
                      scramble_rate=0,
                      scramble_strength=0,
                      replacement_rate=0.2,
                      verbose=True)

plt.plot(fitness_history)
create_chessboard(test_population.population[0], 'integer', size=8)

first = test_population.population[0].genes
board = create_chessboard(test_population.population[0], 'integer', size=8)
print(board)

first = test_population.population[0].genes
board = create_chessboard(test_population.population[0], 'integer', size=8)
print(board)
"""
"""
test_population2 = Population(population_size=100,
                                num_chromosomes=8,
                                chromosome_size=8,
                                representation='binary',
                                num_positives=1)

fitness_history2 = test_population2.train(fitness_function=fitness_non_attacking,
                        num_generations=3000,
                        selection_pressure=0.05,
                        mutation_rate=0.1,
                        mutation_strength=1,
                        mutation_type="replacement",
                        crossover_rate=0.2,
                        crossover_type='ordered_crossover',
                        scramble_rate=0.05,
                        scramble_strength=0.25,
                        replacement_rate=0.3,
                        verbose=True)

create_chessboard(test_population2.population[0], 'binary', size=8)
plt.plot(fitness_history2)

first2 = test_population2.population[0].genes
board2 = create_chessboard(test_population2.population[0], 'binary', size=8)
print(board2)
"""