import random
import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, fitness_func, gene_type, gene_range=None):
        self.fitness_func = fitness_func
        self.gene_type = gene_type
        self.gene_range = gene_range

    #self using version of the generate_genome function
    def generate_genome(self, genome_size, num_positives=None, no_overlap=False):
        self.genome = np.zeros(genome_size, dtype=int)
        
        if self.gene_type == 'binary':
            if num_positives==None:
                num_positives = np.zeros(genome_size)

            positive_indices = np.random.choice(genome_size, num_positives, replace=False)
            self.genome[positive_indices] = 1

        elif self.gene_type == 'integer':
            low, high = self.gene_range

            if no_overlap and (genome_size <= high - low + 1):
                random_values = np.random.choice(np.arange(low, high+1), genome_size, replace=False)
            else:
                random_values = np.random.randint(low, high+1,size=genome_size)
            self.genome = random_values
        
        elif self.gene_type == 'real':
            low, high = self.gene_range if self.gene_range else (-1.0, 1.0)
            self.genome = np.random.uniform(low, high, size=genome_size)
        
        else: raise ValueError("Unknown gene_type: {}".format(self.gene_type))
        return self.genome

    #self using version of the generate_population function
    def generate_population(self, population_size, genome_size, num_positives=None, no_overlap=None,diversity = 0):
        self.population = []
        for _ in range(population_size):
            self.population.append(self.generate_genome(genome_size=genome_size, num_positives=num_positives,no_overlap=no_overlap))
        
        return self.population
    
    #self using version of the mutation function
    def mutation(self, individual, mutation_rate, mutation_strength=1.0):
        for i in range(len(individual)):
                
                if random.random() < mutation_rate:
                    
                    if self.gene_type == 'binary':
                        if random.random() < mutation_rate:
                            # Select two different positions in the individual
                            pos1, pos2 = random.sample(range(len(individual)), 2)
                            # Swap the positions
                            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]


                    elif self.gene_type == 'integer':
                        low, high = self.gene_range
                        individual[i] = random.randint(low, high)

                    elif self.gene_type == 'real':
                        low, high = self.gene_range

                        individual[i] += random.uniform(-mutation_strength, mutation_strength)
                        individual[i] = max(min(individual[i], high), low)
                
        return individual

    #self using version of the crossover function
    def crossover(self, parent1, parent2, crossover_type='one_point'):

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
    
    #self using version of the check_overlap function
    def check_overlap(self, individual):
        if len(individual) == len(set(individual)):
            return False
        else:
            return True
    
    #self using version of the mutate_overlap function
    def mutate_overlap(self, individual):
        while self.check_overlap(individual):
            individual = self.mutation(individual=individual, mutation_rate=0.3, mutation_strength=1.0)
        return individual
    
    #self using version of the population_sort function
    def population_sort(self, population):
        list_of_fitnesses = [(individual, self.fitness_func(individual, self.gene_type)[0]) for individual in population]
        list_of_fitnesses.sort(key=lambda x: x[1], reverse=True)
        return list_of_fitnesses
    
    #self using version of the train function
    def train(self, population, generations, mutation_rate, crossover_type, replacement_rate, mutation_strength, no_overlap, num_positives=None, verbose=True):
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type
        self.replacement_rate = replacement_rate
        self.mutation_strength = mutation_strength
        self.no_overlap = no_overlap
        self.num_positives = num_positives
        self.verbose = verbose

        for gen in range(self.generations):
            self.population = self.population_sort(self.population)

            # convert tuples of genomes to list
            for i in range(len(self.population)):
                self.population[i] = list(self.population[i][0])

            # replace the individuals according to replacement rate
            num_replacements = int(len(self.population)*self.replacement_rate)
            for _ in range(num_replacements):
                self.population.pop()
                self.population.pop()

                # crossover the two most fit individuals
                offspring1, offspring2 = self.crossover(parent1=self.population[0], parent2=self.population[1], crossover_type=self.crossover_type)

                # mutate the offspring
                offspring1 = self.mutation(individual=offspring1, mutation_rate=self.mutation_rate, mutation_strength=self.mutation_strength)
                offspring2 = self.mutation(individual=offspring2, mutation_rate=self.mutation_rate, mutation_strength=self.mutation_strength)

                # check for overlap
                if self.no_overlap:
                    self.mutate_overlap(offspring1)
                    self.mutate_overlap(offspring2)

                #check to see if num_positives is met
                if self.num_positives:
                    if sum(offspring1) != self.num_positives:
                        offspring1 = self.mutation(individual=offspring1, mutation_rate=1, mutation_strength=self.mutation_strength)
                    if sum(offspring2) != self.num_positives:
                        offspring2 = self.mutation(individual=offspring2, mutation_rate=1, mutation_strength=self.mutation_strength)

                self.population.append(offspring1)
                self.population.append(offspring2)

            if self.verbose:
                if gen % (self.generations/100) == 0:
                    print("generation", gen, "individuals" ,len(self.population))
                    print("most fit individual", self.population[0])
                    print("fitness score", self.fitness_func(self.population[0], self.gene_type)[0])
                    print("\n")

        return self.population

