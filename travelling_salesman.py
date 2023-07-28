import random
import numpy as np
import matplotlib.pyplot as plt
from population import *


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

def calculate_path_distance(path, coordinates,flatten=False):

  total_dist = 0

  if flatten:
    path = np.array(path).flatten()

  for i in range(len(path)-1):
    city1 = coordinates[path[i]]
    city2 = coordinates[path[i+1]]
    
    dist = np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    total_dist += dist

  return total_dist

def fitness_travelling_salesman(individual, representation):

  if isinstance(individual, Individual):
    path = individual.genes[0]
  else:
    path = individual

  total_dist = calculate_path_distance(path, coordinates)  
  fitness =  1/total_dist

  return (fitness, total_dist)

'''
coordinates = generate_pc(cities=20, width=100)
plot(coordinates)



# Create test population
test_population = Population(population_size=150,
                              num_chromosomes=1,
                              chromosome_size=20,
                              representation='integer',
                              gene_range=(0, 19),
                              no_overlap=True,
                              num_positives=None)

#train it for a little bit
fitness_history = test_population.train(fitness_function=fitness_travelling_salesman,
                      num_generations=1000,
                      selection_pressure=0.05,
                      mutation_rate=0.2,
                      mutation_type = "replacement",
                      mutation_strength=0.2,
                      crossover_rate=0.2,
                      crossover_type='ordered_crossover',
                      scramble_rate=0.5,
                      scramble_strength=0.25,
                      replacement_rate=0.2,
                      verbose=True)

#fitness of this genome
fitness, distance = fitness_travelling_salesman(test_population.population[0], 'integer')
print("Fitness: ", fitness, "Distance: ", distance)

#plot the fitness history
plt.plot(fitness_history)

'''