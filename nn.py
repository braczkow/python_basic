import random
import math
import copy

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