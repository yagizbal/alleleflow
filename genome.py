import numpy as np
import random

def generate_genome(gene_type, genome_size, gene_range=None,num_positives=None,no_overlap=False):
    genome = np.zeros(genome_size, dtype=int)
    if gene_type == 'binary':
        if num_positives==None:
            num_positives = np.zeros(genome_size)

        positive_indices = np.random.choice(genome_size, num_positives, replace=False)
        genome[positive_indices] = 1

    elif gene_type == 'integer':
        low, high = gene_range

        if no_overlap and (genome_size <= high - low + 1):
            random_values = np.random.choice(np.arange(low, high+1), genome_size, replace=False)
        else:
            random_values = np.random.randint(low, high+1,size=genome_size)
        genome = random_values

    elif gene_type == 'real':
        low, high = gene_range if gene_range else (-1.0, 1.0)
        genome = np.random.uniform(low, high, size=genome_size)

    else: raise ValueError("Unknown gene_type: {}".format(gene_type))
    return genome

def generate_population(population_size, genome_size, gene_type, gene_range=None, num_positives=None, no_overlap=None,diversity = 0):
    population = []
    for _ in range(population_size):
        population.append(generate_genome(gene_type=gene_type, genome_size=genome_size, gene_range=gene_range, num_positives=num_positives,no_overlap=no_overlap))
    
    return population  

def mutation(individual, gene_type, mutation_rate, gene_range=None, mutation_strength=1.0):
    for i in range(len(individual)):
            
            if random.random() < mutation_rate:
                
                if gene_type == 'binary':
                    individual[i] = 1 - individual[i]  # flip the gene

                elif gene_type == 'integer':
                    low, high = gene_range
                    individual[i] = random.randint(low, high)  # assign a new random value within the allowed range
                
                elif gene_type == 'real':
                    low, high = gene_range

                    individual[i] += random.uniform(-mutation_strength, mutation_strength)
                    #mutation strenght is the amount of change that can happen to the gene
                    individual[i] = max(min(individual[i], high), low)

    return individual


def crossover(parent1, parent2, crossover_type='one_point'):
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    rewind1 = parent1
    rewind2 = parent2


    if crossover_type == 'one_point':
        point = random.randint(1, len(parent1)-1)
        offspring1[:point], offspring2[:point] = parent2[:point], parent1[:point]

    elif crossover_type == 'two_point':
        point1 = random.randint(1, len(parent1)-1)
        point2 = random.randint(point1, len(parent1))
        offspring1[point1:point2], offspring2[point1:point2] = parent2[point1:point2], parent1[point1:point2]

    elif crossover_type == 'uniform':
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring1[i], offspring2[i] = parent2[i], parent1[i]

    elif crossover_type == 'ordered':
        start, end = sorted([random.randrange(len(parent1)) for _ in range(2)])
        offspring1[:], offspring2[:] = parent2[:], parent1[:]
        offspring1[start:end], offspring2[start:end] = parent1[start:end], parent2[start:end]
        for i in range(len(parent1)):
            while i < start or i >= end:
                while offspring1[i] in parent1[start:end]:
                    offspring1[i] = parent2[np.where(parent1==offspring1[i])[0][0]]
                while offspring2[i] in parent2[start:end]:
                    offspring2[i] = parent1[np.where(parent2==offspring2[i])[0][0]]
                    i += 1

        else:
            raise ValueError("Unknown crossover_type: {}".format(crossover_type))

    return offspring1, offspring2
    
def check_overlap(individual):
    if len(individual) == len(set(individual)):
        return False
    else:
        return True

def mutate_overlap(individual):
    while check_overlap(individual):
        individual = mutation(individual=individual, gene_type='integer', mutation_rate=0.3, gene_range=(0, 63))
    return individual

def population_sort(population):
    list_of_fitnesses = [(individual, fitness((individual))[0]) for individual in population]
    list_of_fitnesses.sort(key=lambda x: x[1], reverse=True)
    return list_of_fitnesses


def train(population,generations,mutation_rate,crossover_type,replacement_rate,gene_type,gene_range,mutation_strength,verbose=True):
    for gen in range(generations):
        list_of_fitnesses = population_sort(population)


        #convert tuples of genomes to list
        for i in range(len(list_of_fitnesses)):
            list_of_fitnesses[i] = list(list_of_fitnesses[i][0])
        population = list_of_fitnesses

        #according to replacement rate
        for i in range(int(len(population)*replacement_rate)):
            population.pop()
            population.pop()

        #crossover the two most fit individuals
            offspring1,offspring2 = crossover(parent1=population[0], parent2=population[1], crossover_type=crossover_type)

             #mutate the offspring
            offspring1 = mutation(individual=offspring1, gene_type=gene_type, mutation_rate=mutation_rate, gene_range=gene_range, mutation_strength=mutation_strength)
            offspring2 = mutation(individual=offspring2, gene_type=gene_type, mutation_rate=mutation_rate, gene_range=gene_range, mutation_strength=mutation_strength)

            mutate_overlap(offspring1)
            mutate_overlap(offspring2)

            population.append(offspring1)
            population.append(offspring2)

        
        if verbose==True:
            if gen % (generations/100) == 0:
                print("generation",gen)
                print("most fit individual",population[0])
                print("fitness score",fitness((population[0]))[0])
                print("\n")

    return population