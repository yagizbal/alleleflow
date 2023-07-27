from population import *


def create_chessboard(genome, representation, size=8):

    chessboard = np.zeros((size+2, size+2), dtype=int)
    
    if representation == 'binary':

        binary_genome = np.concatenate(genome.genes) # flatten to 1D array
        binary_genome = binary_genome.reshape(size, size) # reshape to 8x8
        
        for i in range(size):
            for j in range(size):
                if binary_genome[i,j] == 1:
                    chessboard[i+1, j+1] = 1

            
    
    elif representation == 'integer':
        
        for i, pos in enumerate(genome.genes):
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

"""
test_population = Population(population_size=100, 
                             num_chromosomes=1,
                             chromosome_size=8, 
                             representation='integer', 
                             gene_range=(0,63),
                             no_overlap=True)

test_population.train(fitness_function=fitness_non_attacking, 
                      num_generations = 10000,
                      selection_pressure=0.05,
                      mutation_rate=0.1,
                        mutation_strength=0.2,
                        mutation_type = "uniform",
                        crossover_rate=0.2,
                        crossover_type='ordered_crossover',
                        replacement_rate=0.3,
                        verbose=True)

print(test_population.population[0].genes)
chessboard = create_chessboard(test_population.population[0], 'integer', size=8)
chessboard = chessboard[1:-1,1:-1]
print(chessboard)
"""
"""
test_population2 = Population(population_size=100,
                                num_chromosomes=8,
                                chromosome_size=8,
                                representation='binary',
                                num_positives=1)

test_population2.train(fitness_function=fitness_non_attacking,
                        num_generations=10000,
                        selection_pressure=0.05,
                        mutation_rate=0.1,
                        mutation_strength=1,
                        mutation_type="replacement",
                        crossover_rate=0.2,
                        crossover_type='ordered_crossover',
                        replacement_rate=0.3,
                        verbose=True)

create_chessboard(test_population2.population[0], 'binary', size=8)

print(test_population2.population[0].genes)
chessboard = create_chessboard(test_population2.population[0], 'binary', size=8)
chessboard = chessboard[1:-1,1:-1]
print(chessboard)
"""