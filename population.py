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


    def remove_duplicates(self):

        genes_ = [str(i.genes) for i in self.population]
        
        unique_genes = set(genes_)

        dict_uniques = {}

        for i in self.population:
            if str(i.genes) in dict_uniques:
                dict_uniques[str(i.genes)].append(i)
            else:
                dict_uniques[str(i.genes)] = [i]

        
        dict_uniques = list(dict_uniques.values())

        lost_unique_genes = []
        for i in dict_uniques:
            lost_unique_genes.append(i[0].genes)
        
        unique_objects_population = []
        for i in dict_uniques:
            unique_objects_population.append(i[0])

        return unique_objects_population, len(self.population) - len(unique_objects_population)


    def train(self,fitness_function, selection_pressure, mutation_rate, mutation_type, mutation_strength, crossover_rate, scramble_rate, scramble_strength,
              crossover_type,replacement_rate, diversity=None, diversity_percent=None, num_generations=None, convergence=None,verbose=True):
        

        if num_generations is None and convergence is None:
            num_generations = 10000
        fitness_history = []

        population_size = len(self.population)
        replace_num = int(population_size * replacement_rate)
        effective_population = int(selection_pressure * population_size) #this value is the number of top individuals that will be used to breed

        if effective_population < 1:
            effective_population = 1

        last = 0


        for generation in range(num_generations):

            fitnesses = []
            for genome in self.population:
                fitnesses.append(fitness_function(individual=genome,representation=self.representation))
           
            if verbose: #print out important stuff 
                percent_completed = int(100 * generation / num_generations)
                if last != percent_completed:

                    print("Percent completed:", percent_completed, "generations", generation)
                    print("Best genome:", self.population[0].genes)
                    print("Best fitness:", np.max(fitnesses), "Lowest fitness:", np.min(fitnesses), "Average:", np.mean(fitnesses),"(standard deviation)", np.std(fitnesses),"\n")

                    fitness_history.append(fitnesses[0][0])
                    last = percent_completed

            if generation %2== 0: #remove duplicates every 2 generations

                pop_, diff = self.remove_duplicates()
                self.population = pop_

                for difference in range(diff):
                    new_individual = Individual(num_chromosomes=self.num_chromosomes, chromosome_size=self.chromosome_size, representation=self.representation, gene_range=self.gene_range, no_overlap=self.no_overlap, num_positives=self.num_positives)
                    new_individual.genome[0].crossover(new_genome1.genome[0])
                    self.population.append(new_individual)


            for replace in range(int(replace_num)): #this part is the breeding, variation and replacement
                zipped = zip(self.population, fitnesses) #this is where the fitnesses are zipped to the population
                zipped = sorted(zipped, key=lambda x: x[1][0], reverse=True) #this is where the population is sorted by fitness
                self.population, fitnesses = zip(*zipped)


                weights = [0.8/effective_population] * effective_population + [0.2/len(self.population)] * (len(self.population) - effective_population) 
                #the way the weights are calculated is that the top 80% of the population will have equal weights, and the bottom 20% will have equal weights

                new_genome1 = copy.deepcopy(random.choices(self.population, weights=weights)[0])
                new_genome2 = copy.deepcopy(random.choices(self.population, weights=weights)[0])

                for gene__ in range(len(new_genome1.genome)):
                    if random.random() < mutation_rate: 
                        new_genome1.genome[gene__].mutate(mutation_type=mutation_type, mutation_strength=mutation_strength) #mutate chromosome

                    if random.random() < crossover_rate:
                        new_genome1.genome[gene__].crossover(new_genome2.genome[gene__]) #crossover chromosome individually

                    if random.random() < scramble_rate:
                        new_genome1.genome[gene__].scramble(scramble_strength=scramble_strength)
                
                self.population = list(self.population)
                self.population = self.population[:-1]
                self.population.append(new_genome1)


        return fitness_history