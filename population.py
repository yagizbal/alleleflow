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
        last = 0


        fitness_history = []


        for generation in range(num_generations):

            fitnesses = []
            for genome in self.population:
                fitnesses.append(fitness_function(individual=genome,representation=self.representation))
            

            if verbose: #print out important stuff 
                percent_completed = int(100 * generation / num_generations)
                if percent_completed > last:
                    print("Percent completed:", percent_completed)
                    print("Best fitness:", fitnesses[0][0])
                    print("Best genome:", self.population[0].genes)
                    print("Min:", np.min(fitnesses[0]), "Max:", np.max(fitnesses[0]), "Avg:", np.mean(fitnesses[0]),"\n")

                    fitness_history.append(fitnesses[0][0])

                    last = percent_completed

            population_size = len(self.population)
            replace_num = int(population_size * replacement_rate)

            if generation %2== 0: #remove duplicates every 2 generations
                dict_uniques = {}
                #individuals with matching genomes are removed, take set of population
                for i in self.population:
                    if str(i.genes) in dict_uniques:
                        dict_uniques[str(i.genes)].append(i)
                    else:
                        dict_uniques[str(i.genes)] = [i]

                pop_, diff = self.remove_duplicates()
                self.population = pop_
                for i in range(diff):
                    new_individual = Individual(num_chromosomes=self.num_chromosomes, chromosome_size=self.chromosome_size, representation=self.representation, gene_range=self.gene_range, no_overlap=self.no_overlap, num_positives=self.num_positives)
                    new_individual.genome[0].crossover(new_genome1.genome[0])
                    self.population.append(new_individual)


            for i in range(int(replace_num)): #this part is the breeding, variation and replacement
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

                new_genome1 = copy.deepcopy(best1)
                new_genome2 = copy.deepcopy(best2)

                #recombinate new_genome1 with new_genome2, note that recombination() is different from crossover()

            
                for i in range(len(new_genome1.genome)):
                    if random.random() < mutation_rate: 
                        #mutate chromosome individually
                        new_genome1.genome[i].mutate(mutation_type=mutation_type, mutation_strength=mutation_strength)

                    if random.random() < crossover_rate:
                        #crossover chromosome individually
                        new_genome1.genome[i].crossover(new_genome2.genome[i])

                    if random.random() < scramble_rate:
                        new_genome1.genome[i].scramble(scramble_strength=scramble_strength)
                
                self.population = list(self.population)



                


                self.population.append(new_genome1)
                self.population = tuple(self.population)



        return fitness_history