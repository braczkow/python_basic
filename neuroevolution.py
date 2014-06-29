import random
import math
import copy

import nn

#random.seed(3)

GENERATIONS = 500
POPULATION = 20

NEURONS = 7


class GeneticAlgorithm:
	def __init__(self, class_type, fitness_function):
		self.class_type = class_type
		self.fitness_function = fitness_function

		self.leader = None

	def _process_generation(self):
		for p in self.current_generation:
			p.fitness = self.fitness_function(p)

		sorted(self.current_generation, key=lambda p: p.fitness)

		leader_candidate = self.current_generation[0]

		if leader_candidate.fitness > self.leader.fitness:
			print "#############################################################################"
			print "Changing leader from ", self.leader.fitness, " to ", leader_candidate.fitness
			self.leader = copy.deepcopy(leader_candidate)
			#print "Leader weights: ", str(self.leader.weights.weights)


		parents = self.current_generation[0 : self.n_population/2]
		#print "Parents lenght: ", len(parents)
		new_generation = []

		while len(parents) > 0:
			parent_a = parents.pop(random.randrange(0, len(parents)))
			parent_b = parents.pop(random.randrange(0, len(parents)))

			new_generation.append(parent_a.reproduce(parent_b))
			new_generation.append(parent_a.reproduce(parent_b))
			new_generation.append(parent_a)
			new_generation.append(parent_b)


		self.current_generation = new_generation



	def run(self, n_generations, n_population):
		self.n_generations = n_generations
		self.n_population = n_population

		self.current_generation = [self.class_type() for i in range(n_population)]
		self.leader = self.current_generation[0]
		self.leader.fitness = -100

		for i in range(0, self.n_generations):
			self._process_generation()

		return self.leader


def neuro_system_impl():
	global NEURONS
	return nn.NeuroSystem(NEURONS, 1)

def xor_fitness(neuro_system):
	inputs =[[-1, -1], [-1, 1], [1, -1], [1, 1]]
	expected = [-1, 1, 1, -1]
	actual = [neuro_system.step(i) for i in inputs]

	diff = [(p - q[0])*(p - q[0]) for p, q in zip(expected, actual)]

	scalar_diff = sum(diff)

	#actual = [neuro_system.step(i) for i in inputs]

	#diff = [(p - q[0])*(p - q[0]) for p, q in zip(expected, actual)]

	#scalar_diff = scalar_diff + sum(diff)

	return (-scalar_diff)



for i in range(10):
	print "=================================Run no. ", i
	ga = GeneticAlgorithm(neuro_system_impl, xor_fitness)
	ga.run(GENERATIONS, POPULATION)
	#print debug
	leader = ga.leader
	leader.print_debug = True

	fitness = xor_fitness(leader)
	print "Leader fitness, default stability_iterations: ", fitness
	#print "Leader weights: ", str(leader.weights.weights)

	leader.stability_iterations = 101

	fitness = xor_fitness(leader)
	print "Leader fitness, nondefault stability_iterations: ", fitness


