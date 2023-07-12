import numpy as np
import random

def generate_genome(gene_type, genome_size, genome_range=None,num_positives=None,no_overlap=False):
    genome = np.zeros(genome_size, dtype=int)

    if gene_type == 'binary':
        if num_positives==None:
            num_positives = np.random.randint(1,genome_size)

        positive_indices = np.random.choice(genome_size, num_positives, replace=False)
        genome[positive_indices] = 1

    elif gene_type == 'integer':
        low, high = genome_range

        if no_overlap and (genome_size <= high - low + 1):
            random_values = np.random.choice(np.arange(low, high+1), genome_size, replace=False)
        else:
            random_values = np.random.randint(low, high+1)
        genome = random_values

    elif gene_type == 'real':
        low, high = genome_range if genome_range else (-1.0, 1.0)
        genome = np.random.uniform(low, high, size=genome_size)

    else:
        raise ValueError("Unknown gene_type: {}".format(gene_type))
    return genome

def initialize_population(population_size,num_genes,gene_type,gene_range=None,num_positives=None):
    population = [generate_genome(gene_type, num_genes, gene_range,num_positives) for _ in range(population_size)]
    return population

def mutation(individual, gene_type, mutation_rate, gene_range=None):
    for i in range(len(individual)):
            if random.random() < mutation_rate:
                if gene_type == 'binary':
                    individual[i] = 1 - individual[i]  # flip the gene
                elif gene_type == 'integer':
                    low, high = gene_range
                    individual[i] = random.randint(low, high)  # assign a new random value within the allowed range

    return individual

def crossover(parent1, parent2, crossover_type='one_point'):
    offspring1, offspring2 = parent1.copy(), parent2.copy()
        
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
    