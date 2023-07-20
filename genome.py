import random
import numpy as np 
import copy 

class Chromosome:
    def __init__(self, chromosome_type, gene_type, gene_range=None, no_overlap=False):
        self.chromosome_type = chromosome_type
        self.gene_type = gene_type
        self.gene_range = gene_range
        self.no_overlap = no_overlap
        self.genes = None


    def generate_chromosome(self, chromosome_size, num_positives=None):
        self.genes = np.zeros(chromosome_size, dtype=int)
        
        
        if self.gene_type == 'binary':
            if num_positives is None:
                num_positives = np.random.randint(chromosome_size + 1)  # number of 1s can be between 0 and chromosome_size

            self.genes = np.zeros(chromosome_size, dtype=int)
            positive_indices = np.random.choice(chromosome_size, num_positives, replace=False)
            self.genes[positive_indices] = 1

        elif self.gene_type == 'integer':
            low, high = self.gene_range

            if self.no_overlap and (chromosome_size <= high - low + 1):
                random_values = np.random.choice(np.arange(low, high+1), chromosome_size, replace=False)
            else:
                random_values = np.random.randint(low, high+1,size=chromosome_size)
            self.genes = random_values
        
        elif self.gene_type == 'real':
            low, high = self.gene_range if self.gene_range else (-1.0, 1.0)
            self.genes = np.random.uniform(low, high, size=chromosome_size)
        
        else: raise ValueError("Unknown gene_type: {}".format(self.gene_type))
        return self.genes

    def mutate(self, mutation_strength=1.0):                    
        num_mutations = int(mutation_strength * len(self.genes))
        mutation_indices = random.sample(range(len(self.genes)), num_mutations)

        if self.gene_type == 'binary':
            self.genes[mutation_indices] = 1 - self.genes[mutation_indices]
        elif self.gene_type == 'integer':
            low, high = self.gene_range
            self.genes[mutation_indices] = np.random.randint(low, high+1, size=num_mutations)
        elif self.gene_type == 'real':
            low, high = self.gene_range
            self.genes[mutation_indices] = np.random.uniform(low, high, size=num_mutations)


    def crossover(self, other, crossover_rate, no_overlap, crossover_type='single_point'):
        #initial values as backups in case you need to revert

        if random.random() < crossover_rate:
            if crossover_type == 'single_point':
                crossover_point = random.randint(1, len(self.genes) - 1)
                self.genes = np.concatenate((self.genes[:crossover_point], other.genes[crossover_point:]))
                other.genes = np.concatenate((other.genes[:crossover_point], self.genes[crossover_point:]))
                
            elif crossover_type == 'two_point':
                crossover_points = sorted(random.sample(range(1, len(self.genes) - 1), 2))
                self.genes = np.concatenate((self.genes[:crossover_points[0]], other.genes[crossover_points[0]:crossover_points[1]], self.genes[crossover_points[1]:]))
                other.genes = np.concatenate((other.genes[:crossover_points[0]], self.genes[crossover_points[0]:crossover_points[1]], other.genes[crossover_points[1]:]))
                
            elif crossover_type == 'uniform':
                for i in range(len(self.genes)):
                    if random.random() < 0.5:
                        self.genes[i], other.genes[i] = other.genes[i], self.genes[i]
            else:
                raise ValueError("Unknown crossover type: {}".format(crossover_type))
        




    def check_overlap(self):
        return len(set(self.genes)) != len(self.genes)
        #this will return true if there are duplicates in the chromosome


class Genome:
    def __init__(self, num_chromosomes, chromosome_size, representation, gene_range=None, no_overlap=False, num_positives=None):
        self.chromosomes = {i: Chromosome(i, representation, gene_range, no_overlap) for i in range(num_chromosomes)}
        self.num_chromosomes = num_chromosomes
        self.chromosome_size = chromosome_size
        self.representation = representation
        self.gene_range = gene_range
        self.no_overlap = no_overlap
        self.num_positives = num_positives
        
        for chromosome in self.chromosomes.values():
            chromosome.generate_chromosome(chromosome_size, num_positives)
        self.update_genes()

    def mutate(self, mutation_rate, mutation_strength=1.0):
        #pick a random chromosome to mutate
        chromosome = random.choice(list(self.chromosomes.values()))
        chromosome.mutate(mutation_strength)
        self.update_genes()

    def crossover(self, other_genome, crossover_rate, no_overlap, crossover_type='single_point'):
        for i in range(self.num_chromosomes):
            self.chromosomes[i].crossover(other_genome.chromosomes[i], no_overlap, crossover_rate, crossover_type,)
        self.update_genes()
        other_genome.update_genes()


    def recombination(self, other_genome, pieces=2):
        # Create new genomes
        new_genome1 = Genome(num_chromosomes=len(self.chromosomes), 
                            chromosome_size=len(self.chromosomes[0].genes),  
                            representation=self.chromosomes[0].gene_type,
                            gene_range=self.chromosomes[0].gene_range,
                            no_overlap=self.chromosomes[0].no_overlap)

        new_genome2 = Genome(num_chromosomes=len(self.chromosomes),
                            chromosome_size=len(self.chromosomes[0].genes),
                            representation=self.chromosomes[0].gene_type,
                            gene_range=self.chromosomes[0].gene_range,
                            no_overlap=self.chromosomes[0].no_overlap)

        # Determine split point
        mid = len(self.chromosomes) // pieces
        
        # Recombine first half
        for i in range(mid):
            new_genome1.chromosomes[i] = self.chromosomes[i]
            new_genome2.chromosomes[i] = other_genome.chromosomes[i]

        # Recombine second half
        for i in range(mid, len(self.chromosomes)):
            new_genome1.chromosomes[i] = other_genome.chromosomes[i] 
            new_genome2.chromosomes[i] = self.chromosomes[i]


        # Return new genomes
        return new_genome1, new_genome2

    def update_genes(self):
        self.genes = {i: chromosome.genes for i, chromosome in self.chromosomes.items()}

    def check_overlap(self):
        for chromosome in self.chromosomes.values():
            if chromosome.check_overlap():
                return True
        return False
        

class Population:
    def __init__(self, population_size, num_chromosomes, chromosome_size, gene_type, gene_range=None, no_overlap=False, num_positives=None):
        #self.population = [Genome(num_chromosomes, chromosome_size, gene_type, gene_range, no_overlap, num_positives) for i in range(population_size)]
        self.population_size = population_size
        self.num_chromosomes = num_chromosomes
        self.chromosome_size = chromosome_size
        self.representation = gene_type
        self.gene_range = gene_range
        self.no_overlap = no_overlap
        self.num_positives = num_positives

        self.population = [Genome(self.num_chromosomes, self.chromosome_size, self.representation, self.gene_range, self.no_overlap, self.num_positives) for i in range(self.population_size)]

    def train(self, fitness_function, num_generations, mutation_rate, mutation_strength, crossover_rate, crossover_type,replacement_rate,verbose=True):
        '''Population -> Population to be trained
        fitness_function -> function that takes in a genome and returns a fitness value, specific to the problem
        num_generations -> number of generations to train for
        mutation_rate -> probability of any given individual of the population undergoing a random mutation
        mutation_strength -> how much to mutate by
        crossover_rate -> probability of individuals undergoing replacement undergoing a crossover
        crossover_type -> type of crossover to use
        replacement_rate -> probability of individuals undergoing replacement
        '''

        for generation in range(num_generations):

            fitnesses = []
            for genome in self.population:
                fitnesses.append(fitness_function(individual=genome,representation='integer'))

            population_size = len(self.population)
            replace_num = int(population_size * replacement_rate)

            for i in range(int(replace_num/2)): #this part is the "breeding" and replacement
                zipped = zip(self.population, fitnesses) 
                zipped = sorted(zipped, key=lambda x: x[1][0], reverse=True)
                self.population, fitnesses = zip(*zipped)


                self.population = self.population[:-2]
                #print the remaining genomes of the population

                best1 = self.population[0]
                best2 = self.population[1]

                # Recombine into two new genomes
                new_genome1 = Genome(num_chromosomes=best1.num_chromosomes, chromosome_size=best1.chromosome_size, representation=best1.representation, gene_range=best1.gene_range, no_overlap=best1.no_overlap, num_positives=best1.num_positives)
                new_genome2 = Genome(num_chromosomes=best2.num_chromosomes, chromosome_size=best2.chromosome_size, representation=best2.representation, gene_range=best2.gene_range, no_overlap=best2.no_overlap, num_positives=best2.num_positives)

                backup1 = copy.deepcopy(new_genome1)
                backup2 = copy.deepcopy(new_genome2)

                
                if random.random() < mutation_rate:
                    new_genome1.mutate(mutation_strength)
                if random.random() < mutation_rate:
                    new_genome2.mutate(mutation_strength)

                if random.random() < crossover_rate: #crossover function from chromosome
                    new_genome1.crossover(other_genome=new_genome2, no_overlap=self.no_overlap, crossover_rate=crossover_rate, crossover_type=crossover_type)
                
                if new_genome1.check_overlap():
                    #print genes
                    #print("ab",new_genome1.genes,backup1.genes)
                    new_genome1.genes = backup1.genes
                    #print(new_genome1.genes)
                    #print("\n")
                if new_genome2.check_overlap():
                    #print("ba",new_genome2.genes,backup2.genes)
                    new_genome2.genes = backup2.genes

                #add the new genomes to the population tuple
                self.population = self.population + (new_genome1,new_genome2)
                    
            if verbose==True and generation % 100 == 0:
                print("Generation: ", generation)
                print("Best fitness: ", fitnesses[0][0])
                print("Best genome: ", self.population[0].genes)
                print("\n")
