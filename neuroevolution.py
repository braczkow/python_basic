import random
import math
import copy

#random.seed(3)

GENERATIONS = 500
POPULATION = 20

NEURONS = 7

MUTATION_RATE = 0.02
MUTATION_MEAN = 0
MUTATION_SIGMA = 0.5

STABILITY_LEVEL = 0.01

class Neuron:
	def __init__(self, N, id):
		self.N = N
		self.weights = []
		self.id = id
		self.value = 0
		self.beta = 5
		self.catchedInput = 0
		for i in range (0, N):
			self.weights.append(random.uniform(-1, 1));
		
		self.weights[id] = 0
		
	def __str__(self):
		return "id: " + str(self.id) + " value: " + str(self.value) + "(" + '; '.join([str(w) for w in self.weights]) + ")"

	def catchInputs(self, current_state):
		self.catchedInput = sum(p*q for p,q in zip(self.weights, current_state))

	def update(self):
		self.value = 1/(1 + math.exp(-1 * self.beta * self.catchedInput))

	def mutate(self):
		global MUTATION_RATE
		global MUTATION_MEAN
		global MUTATION_SIGMA

		for i in range (0, self.N):
			rand = random.uniform(0, 1)
			if rand < MUTATION_RATE:
				change = random.normalvariate(MUTATION_MEAN, MUTATION_SIGMA)
				self.weights[i] = self.weights[i] + change

		self.weights[self.id] = 0


class NeuroSystem:
	def __init__(self, N, n_to_return):
		self.N = N
		self.n_to_return = n_to_return
		self.neurons = []
		for i in range(0, N):
			self.neurons.append(Neuron(N, i));
			
	def __str__(self):
		return "N: " + str(self.N) +'\n' + ', '.join([str(n) for n in self.neurons])

	def _advance(self):	
		current_state = [n.value for n in self.neurons];
			
		for i in range(0, self.N):
			self.neurons[i].catchInputs(current_state);

		for i in range(0, self.N):
			self.neurons[i].update();


		
	def step(self, inputs):		
		if len(inputs) > self.N:
			print "len(inputs) > self.N"
			return
		
		for i in range(0, len(inputs)):
			self.neurons[i].value = inputs[i]

		#current_state = [n.value for n in self.neurons];
		#print current_state	

		#TODO add stabilization
		i = 0
		not_stable = True
		while i < 100 and not_stable:
			current_state = [n.value for n in self.neurons];

			self._advance()
			
			new_state = [n.value for n in self.neurons];

			diff = [(p-q)*(p-q) for p, q in zip(current_state, new_state)]
			scalar_diff = sum(diff)

			global STABILITY_LEVEL

			if scalar_diff < STABILITY_LEVEL:
				not_stable = False

			i = i+1

		#current_state = [n.value for n in self.neurons];
		#print current_state	

		return [self.neurons[i].value for i in range(self.N - self.n_to_return, self.N)]

	def _cross(self, partner):
		if self.N != partner.N:
			print "We cannot DO it!"
			return

		neurons = []
		for i in range(0, self.N):
			toss = bool(random.getrandbits(1))
			if (toss):
				neurons.append(self.neurons[i])
			else:
				neurons.append(partner.neurons[i])

		offspring = NeuroSystem(self.N, self.n_to_return)
		offspring.neurons = neurons

		return offspring

	def _mutate(self):
		[n.mutate() for n in self.neurons]


	def evolve(self, partner):
		offspring = self._cross(partner)
		offspring._mutate()

		return offspring


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
			print "Changed leader from ", self.leader.fitness, " to ", leader_candidate.fitness
			self.leader = copy.deepcopy(leader_candidate)


		parents = self.current_generation[0 : self.n_population/2]
		#print "Parents lenght: ", len(parents)
		new_generation = []

		while len(parents) > 0:
			parent_a = parents.pop(random.randrange(0, len(parents)))
			parent_b = parents.pop(random.randrange(0, len(parents)))

			new_generation.append(parent_a.evolve(parent_b))
			new_generation.append(parent_a.evolve(parent_b))
			new_generation.append(parent_a)
			new_generation.append(parent_b)


		self.current_generation = new_generation



	def run(self, n_generations, n_population):
		self.n_generations = n_generations
		self.n_population = n_population

		self.current_generation = [self.class_type() for i in range(n_population)]
		self.leader = self.current_generation[0]
		self.leader.fitness = 0

		for i in range(0, self.n_generations):
			self._process_generation()

		return self.leader


def neuro_system_impl():
	global NEURONS
	return NeuroSystem(NEURONS, 1)

def xor_fitness(neuro_system):
	inputs =[[0, 0], [0, 1], [1, 0], [1, 1]]
	expected = [0, 1, 1, 0]
	actual = [neuro_system.step(i) for i in inputs]

	diff = [(p - q[0])*(p - q[0]) for p, q in zip(expected, actual)]

	scalar_diff = sum(diff)

	return 4 - scalar_diff

def xor_fitness_debug(neuro_system):
	inputs =[[0, 0], [0, 1], [1, 0], [1, 1]]
	expected = [0, 1, 1, 0]
	actual = [neuro_system.step(i) for i in inputs]

	print "actual: " + str(actual)

	diff = [(p - q[0])*(p - q[0]) for p, q in zip(expected, actual)]

	scalar_diff = sum(diff)

	return 4 - scalar_diff

for i in range(10):
	print "Run no. ", i
	ga = GeneticAlgorithm(neuro_system_impl, xor_fitness)
	ga.run(GENERATIONS, POPULATION)
