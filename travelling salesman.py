import random
import numpy as np
import matplotlib.pyplot as plt
from genome import *

def generate_space(points,width):
    ls = []
    for x in range(0,points):
        a = [random.randrange(1,width+1),random.randrange(1,width+1)] #you need to make it so that it doesnt generate two on the same spot
        ls.append(a)
    return (np.array(ls))

def plot(coordinates):
    xs = []
    ys = []
    for x in coordinates:
        xs.append(x[0]) #doesnt really matter which you take as x and which you take as y
        ys.append(x[1])
    plt.scatter(xs,ys)
    for i,ğ in enumerate(coordinates):
        #plt.annotate(i,(xs[i],ys[i]))
        plt.annotate(f"{i} ({xs[i]},{ys[i]})",(xs[i],ys[i]))
    plt.gcf().set_size_inches(10, 10)
    plt.show()

def generate_pc(cities,width):
    coordinates = generate_space(cities,width)
    return coordinates

def calculate_path_distance(path, coordinates):

  total_dist = 0

  for i in range(len(path)-1):
    city1 = coordinates[path[i]]
    city2 = coordinates[path[i+1]]
    
    dist = np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    total_dist += dist

  return total_dist

def fitness_travelling_salesman(individual, representation):

  if isinstance(individual, Genome):
    path = individual.genes[0]
  else:
    path = individual

  total_dist = calculate_path_distance(path, coordinates)  
  fitness =  1/total_dist

  return (fitness, total_dist)


  # Generate 10 city TSP
coordinates = generate_pc(cities=50, width=50) 

# Create test population
test_population = Population(population_size=50,
                              num_chromosomes=1,
                              chromosome_size=50,
                              gene_type='integer',
                              gene_range=(0, 9),
                              no_overlap=True,
                              num_positives=None)
#train it for a little bit
test_population.train(fitness_function=fitness_travelling_salesman,
                      num_generations=10000,
                      selection_pressure=0.5,
                      mutation_rate=0.5,
                      mutation_strength=0.5,
                      crossover_rate=0.5,
                      crossover_type='two_point',
                      replacement_rate=0.5,
                      verbose=True)

#fitness of this genome
fitness, distance = fitness_travelling_salesman(test_population.population[0], 'integer')
print("Fitness: ", fitness, "Distance: ", distance)
