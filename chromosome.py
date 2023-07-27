import random
import numpy as np


class Chromosome:
    def __init__(self, chromosome_size, gene_type, gene_range=None, no_overlap=False,num_positives=None):
        self.gene_type = gene_type
        self.gene_range = gene_range
        self.no_overlap = no_overlap
        self.chromosome_size = chromosome_size
        self.num_positives = num_positives
        self.genes = None


        self.genes = np.zeros(chromosome_size, dtype=int)
        
        if self.gene_type == 'binary':
            if num_positives is None:
                num_positives = np.random.randint(chromosome_size + 1)  #randomly assign number of positives if not specified

            self.genes = np.zeros(chromosome_size, dtype=int)
            positive_indices = np.random.choice(chromosome_size, num_positives, replace=False)
            self.genes[positive_indices] = 1

        elif self.gene_type == 'integer':
            low, high = self.gene_range #unpack tuple, low and high are the bounds

            if self.no_overlap and (chromosome_size <= high - low + 1):
                random_values = np.random.choice(np.arange(low, high+1), chromosome_size, replace=False)
            else:
                random_values = np.random.randint(low, high+1,size=chromosome_size)
            self.genes = random_values
        
        #elif self.gene_type == 'real':
        #    low, high = self.gene_range if self.gene_range else (-1.0, 1.0)
        #    self.genes = np.random.uniform(low, high, size=chromosome_size)
        
        else: raise ValueError("Unknown gene_type: {}".format(self.gene_type))

    def mutate(self, mutation_type, mutation_strength=1.0):
        if mutation_type == "replacement":
            #swap the values of two random indices
            index1 = random.randint(0, len(self.genes) - 1)
            index2 = random.randint(0, len(self.genes) - 1)
            self.genes[index1], self.genes[index2] = self.genes[index2], self.genes[index1]

        elif mutation_type == "uniform":
            #bounds of mutation are dependant on gene type
            
            if self.gene_type == 'binary':
                available_values = [0,1]
                mutation_range = (0, 1)


                if self.num_positives:
                    for i in range(self.num_positives):
                        self.mutate('replacement')
                else:
                    amount = len(self.genes)*mutation_strength
                    for i in range(int(amount)):
                        index = random.randint(0, len(self.genes) - 1)
                        value = random.choice(mutation_range)
                        self.genes[index] = value
            
            elif self.gene_type in ('integer','real'):
                #if no_overlap is true, remove existing values from the list of possible values for mutation
                if self.no_overlap:
                    mutation_range = self.gene_range
                    #tupleX = [x for x in tupleX if x > 5]
                    unavailable_values = set(self.genes)
                    all_values = set(range(*self.gene_range))
                    available_values = list(all_values - unavailable_values)
                    mutation_range = tuple(available_values)

                else:
                    mutation_range = self.gene_range
                    
                #mutate in accordance with mutation strength
                amount =len(self.genes)*mutation_strength
                for i in range(int(amount)):
                    index = random.randint(0, len(self.genes) - 1)
                    value = random.choice(mutation_range)
                    self.genes[index] = value

                    if self.no_overlap:      
                        available_values = list(set(available_values) - set([self.genes[index]]))
                        mutation_range = tuple(available_values)
           
    def crossover(self, other, crossover_type='single_point'):

        '''
        other: another chromosome instance, this is the chromosome that will be crossed over with self
        crossover_type: 'single_point' or 'two_point', this determines how the chromosomes are crossed over
        '''

        #initial values as backups in case you need to revert
        backup_self = np.copy(self.genes)
        backup_other = np.copy(other.genes)

        if crossover_type == 'single_point':
            index_of_crossover = random.randint(1, len(self.genes) - 1)

            piece1 = self.genes[:index_of_crossover]
            piece2 = other.genes[index_of_crossover:]

            piece3 = other.genes[:index_of_crossover]
            piece4 = self.genes[index_of_crossover:]

            self.genes = np.concatenate((piece1, piece2))
            other.genes = np.concatenate((piece3, piece4))

        elif crossover_type == 'ordered_crossover':
            val1 = random.randrange(1,len(other.genes)-1)

            random_indices1 = np.random.choice(len(other.genes), val1,replace=False)
            random_values1 = other.genes[random_indices1]

            set1 = list(set(random_values1))
            set2 = list(set(self.genes) )

            ls = [[],[]]
            for i in set1:
                if i not in set2:
                    ls[0].append(i)
                    #append the index of the element in the other genome to the other list
                    ls[1].append(np.where(other.genes == i)[0][0])

            for i in range(len(ls[0])):
                self.genes[ls[1][i]] = ls[0][i]
            
    def check_overlap(self):
        return len(set(self.genes)) != len(self.genes)
        #this will return true if there are duplicates in the chromosome

'''
for i in range(10):
    chromosome1 = Chromosome(chromosome_size=10, gene_type='binary', num_positives=4)
    print("generated", chromosome1.genes, type(chromosome1.genes), sum(chromosome1.genes))
    chromosome1.mutate(mutation_type='uniform', mutation_strength=0.3)
    print("mutated  ",chromosome1.genes, type(chromosome1.genes), "sum", sum(chromosome1.genes))
    print("\n")

#no specified number of positives
for i in range(10):
    chromosome1 = Chromosome(chromosome_size=10, gene_type='binary')
    print("generated", chromosome1.genes, type(chromosome1.genes), sum(chromosome1.genes))
    chromosome1.mutate(mutation_type='uniform', mutation_strength=0.5)
    print("mutated  ",chromosome1.genes, type(chromosome1.genes), "sum", sum(chromosome1.genes))
    print("\n")

#for i in range(10):
    chromosome2 = Chromosome(chromosome_size=10, gene_type='integer', gene_range=(1,15), no_overlap=True)
    print("generated", chromosome2.genes, type(chromosome2.genes), chromosome2.check_overlap())
    chromosome2.mutate(mutation_type='uniform', mutation_strength=0.3)
    print("mutated  ",chromosome2.genes, type(chromosome2.genes), "overlap", chromosome2.check_overlap())

    print("\n")

#with overlap
for i in range(10):
    chromosome2 = Chromosome(chromosome_size=10, gene_type='integer', gene_range=(1,15), no_overlap=False)
    print("generated", chromosome2.genes, type(chromosome2.genes), chromosome2.check_overlap())
    chromosome2.mutate(mutation_type='uniform', mutation_strength=0.3)
    print("mutated  ",chromosome2.genes, type(chromosome2.genes), "overlap", chromosome2.check_overlap())
    print("\n")

#crossover check
for i in range(10):
    chromosome1 = Chromosome(chromosome_size=10, gene_type='integer', gene_range=(1,100), no_overlap=True)
    chromosome2 = Chromosome(chromosome_size=10, gene_type='integer', gene_range=(1,100), no_overlap=True)
    print("chromosome 1 ", chromosome1.genes, type(chromosome1.genes), chromosome1.check_overlap())
    print("chromosome 2 ", chromosome2.genes, type(chromosome2.genes), chromosome2.check_overlap())

    chromosome1.crossover(chromosome2, crossover_type='ordered_crossover')
    print("crossovered 1", chromosome1.genes, type(chromosome1.genes), chromosome1.check_overlap(),"\n")
'''