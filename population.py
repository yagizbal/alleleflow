from individual import *
import numpy as np
import random
import copy

class Population:
    def __init__(self,population_size,num_chromosomes,chromosome_size,representation,gene_range=None,no_overlap=False,num_positives=None):
        self.population_size = population_size
        self.num_chromosomes = num_chromosomes
        self.chromosome_size = chromosome_size
        self.representation = representation
        self.gene_range = gene_range
        self.no_overlap = no_overlap
        self.num_positives = num_positives

        self.population = []
        for i in range(population_size):
            self.population.append(Individual(num_chromosomes=num_chromosomes, chromosome_size=chromosome_size, representation=representation, gene_range=gene_range, no_overlap=no_overlap, num_positives=num_positives))

    def train(self,fitness_function, selection_pressure, mutation_rate, mutation_type, mutation_strength, crossover_rate, crossover_type,replacement_rate, num_generations=None, convergence=None,verbose=True):
        if num_generations is None and convergence is None:
            num_generations = 10000
        last = 0

        for generation in range(num_generations):


            if verbose:
                percent_completed = int(100 * generation / num_generations)
                if percent_completed > last:
                    print("Percent completed:", percent_completed)
                    print("Best fitness:", fitnesses[0][0])
                    print("Best genome:", self.population[0].genes)
                    print("length of population", len(self.population))

                    last = percent_completed

            fitnesses = []
            for genome in self.population:
                fitnesses.append(fitness_function(individual=genome,representation=self.representation))


            population_size = len(self.population)
            replace_num = int(population_size * replacement_rate)

            for i in range(int(replace_num)): #this part is the "breeding" and replacement
                zipped = zip(self.population, fitnesses) 
                zipped = sorted(zipped, key=lambda x: x[1][0], reverse=True)
                self.population, fitnesses = zip(*zipped)


                self.population = self.population[:-1]
                num_top = int(selection_pressure * population_size) #this value is the number of top individuals that will be used to breed

                if num_top < 1:
                    num_top = 1

                weights = [0.8/num_top] * num_top + [0.2/len(self.population)] * (len(self.population) - num_top)

                best1 = random.choices(self.population, weights=weights)[0]
                best2 = random.choices(self.population, weights=weights)[0]

                # Recombine into two new genomes
                new_genome1 = Individual(num_chromosomes=best1.num_chromosomes, chromosome_size=best1.chromosome_size, 
                                         representation=best1.representation, gene_range=best1.gene_range, no_overlap=best1.no_overlap, num_positives=best1.num_positives)
                new_genome2 = Individual(num_chromosomes=best2.num_chromosomes, chromosome_size=best2.chromosome_size,
                                          representation=best2.representation, gene_range=best2.gene_range, no_overlap=best2.no_overlap, num_positives=best2.num_positives)

                backup1 = copy.deepcopy(new_genome1)
                backup2 = copy.deepcopy(new_genome2)

                #recombinate new_genome1 with new_genome2, note that recombination() is different from crossover()
                new_genome1.recombination(new_genome2)
            
                for i in range(len(new_genome1.genome)):
                    if random.random() < mutation_rate: 
                        #mutate chromosome individually
                        new_genome1.genome[i].mutate(mutation_type=mutation_type, mutation_strength=mutation_strength)

                    if random.random() < crossover_rate:
                        #crossover chromosome individually
                        new_genome1.genome[i].crossover(new_genome2.genome[i])
                
                #recombination
                

                #add new genome to population
                #self.population.append(new_genome1) #this is a tuple so you cannot append
                #here is how you do it instead
                self.population = list(self.population)
                self.population.append(new_genome1)
                self.population = tuple(self.population)

