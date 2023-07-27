from chromosome import *
import random
import numpy as np

class Individual:
    def __init__(self, num_chromosomes, chromosome_size, representation, gene_range=None, no_overlap=False, num_positives=None):

        self.genome = {}
        for i in range(num_chromosomes):
            self.genome[i] = Chromosome(chromosome_size=chromosome_size, gene_type=representation, gene_range=gene_range, no_overlap=no_overlap, num_positives=num_positives)

        self.genes = [v.genes for v in self.genome.values()]
        
        
        self.num_chromosomes = num_chromosomes
        self.chromosome_size = chromosome_size
        self.representation = representation
        self.gene_range = gene_range
        self.no_overlap = no_overlap
        self.num_positives = num_positives

    def recombination(self, other_genome, pieces=2):
        # Create new genomes
        new_genome1 = Individual(num_chromosomes=len(self.genome), 
                            chromosome_size=len(self.genome[0].genes),  
                            representation=self.genome[0].gene_type,
                            gene_range=self.genome[0].gene_range,
                            no_overlap=self.genome[0].no_overlap)


        num_chromosomes = len(self.genome)
        num_swap = num_chromosomes // pieces
        swap_indices = random.sample(range(num_chromosomes), num_swap)

        for i in range(num_chromosomes):
            if i in swap_indices:
                new_genome1.genome[i] = self.genome[i]
            else:
                new_genome1.genome[i] = other_genome.genome[i]

        self.genome = new_genome1.genome
        self.genes = [v.genes for v in self.genome.values()]

    def genomic_overlap(self):
        overlaps = 0
        for i in range(len(self.genes)):
            for j in range(i+1, len(self.genes)):
                if len(set(self.genes[i]) & set(self.genes[j])) > 0:
                    overlaps += 1
        return overlaps

"""
#generate 10 pops and check them for overlap
for i in range(10):
    pop1 = Individual(num_chromosomes=2, chromosome_size=4, representation='integer', gene_range=(1,10), no_overlap=True)
    print("Genome:", pop1.genes, "Genomic overlap:", pop1.genomic_overlap())


#10 recombination
for i in range(10):
    pop1 = Individual(num_chromosomes=6, chromosome_size=4, representation='integer', gene_range=(1,10), no_overlap=True)
    pop2 = Individual(num_chromosomes=6, chromosome_size=4, representation='integer', gene_range=(11,20), no_overlap=True)
    print("Genome 1:", pop1.genes, "overlap value", pop1.genomic_overlap(), "\nGenome 2:", pop2.genes, "overlap value", pop2.genomic_overlap())

    pop1.recombination(pop2)
    print("Recombined 1:", pop1.genes, "overlap value", pop1.genomic_overlap(),"\n")
"""