import random
import math
import copy

import nn
import MLP

#random.seed(3)

GENERATIONS = 3000
POPULATION = 20

class GeneticAlgorithm:
	def __init__(self, neuronet_type, evolutor_type, fitness_function):
		self.neuronet_type = neuronet_type
		self.evolutor_type = evolutor_type
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

			evolutor = self.evolutor_type(parent_a, parent_b)

			new_generation.append(evolutor.reproduce())
			new_generation.append(evolutor.reproduce())
			new_generation.append(parent_a)
			new_generation.append(parent_b)


		self.current_generation = new_generation



	def run(self, n_generations, n_population):
		self.n_generations = n_generations
		self.n_population = n_population

		self.current_generation = [self.neuronet_type() for i in range(n_population)]
		self.leader = self.current_generation[0]
		self.leader.fitness = -100

		for i in range(0, self.n_generations):
			self._process_generation()

		return self.leader


def neuro_net_impl():
	impl = MLP.MLPNetwork(3, [2, 4, 1])
	return impl

def neuro_evolutor_impl(lhs, rhs):
	return MLP.MLPNetworkEvolutor(lhs, rhs)

def xor_fitness(neuro_system, print_debug = False):
	inputs =[[-1, -1], [-1, 1], [1, -1], [1, 1]]
	expected = [-1, 1, 1, -1]
	actual = [neuro_system.step(i) for i in inputs]

	if print_debug == True:
		print "Actual: " + str(actual)

	diff = [(p - q[0])*(p - q[0]) for p, q in zip(expected, actual)]

	scalar_diff = sum(diff)

	#actual = [neuro_system.step(i) for i in inputs]

	#diff = [(p - q[0])*(p - q[0]) for p, q in zip(expected, actual)]

	#scalar_diff = scalar_diff + sum(diff)

	return (-scalar_diff)



for i in range(10):
	print "=================================Run no. ", i
	ga = GeneticAlgorithm(neuro_net_impl, neuro_evolutor_impl, xor_fitness)
	ga.run(GENERATIONS, POPULATION)
	#print debug
	leader = ga.leader
	leader.print_debug = True

	fitness = xor_fitness(leader, True)
	print "Leader fitness, default stability_iterations: ", fitness
	#print "Leader weights: ", str(leader.weights.weights)


