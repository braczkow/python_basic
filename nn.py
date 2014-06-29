import random
import math
import copy

MUTATION_RATE = 0.02
MUTATION_MEAN = 0
MUTATION_SIGMA = 0.5

STABILITY_LEVEL = 0.01
STABILITY_ITERATIONS = 100

class Neuron:
	def __init__(self, N, id):
		self.N = N
		self.weights = []
		self.id = id
		self.value = 0
		self.beta = 5
		self.catched_input = 0

		for i in range (0, N):
			self.weights.append(random.uniform(-1, 1));
		
		self.weights[id] = 0
		
	def __str__(self):
		return "id: " + str(self.id) + " value: " + str(self.value) + "(" + '; '.join([str(w) for w in self.weights]) + ")"

	def catch_inputs(self, current_state):
		self.catched_input = sum(p*q for p,q in zip(self.weights, current_state))

	def update(self):
		self.value = 1/(1 + math.exp(-1 * self.beta * self.catched_input))

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
		self.print_debug = False

		global STABILITY_LEVEL
		global STABILITY_ITERATIONS

		self.stability_level = STABILITY_LEVEL
		self.stability_iterations = STABILITY_ITERATIONS


		for i in range(0, N):
			self.neurons.append(Neuron(N, i));
			
	def __str__(self):
		return "N: " + str(self.N) +'\n' + ', '.join([str(n) for n in self.neurons])

	def _advance(self):	
		current_state = [n.value for n in self.neurons];
			
		for i in range(0, self.N):
			self.neurons[i].catch_inputs(current_state);

		for i in range(0, self.N):
			self.neurons[i].update();

	def _stabilize(self):


		i = 0
		not_stable = True
		scalar_diff = 0
		return_diff = 0
		while i < self.stability_iterations and not_stable:
			current_state = [n.value for n in self.neurons];
			self._advance()
			new_state = [n.value for n in self.neurons];

			diff = [(p-q)*(p-q) for p, q in zip(current_state, new_state)]
			scalar_diff = sum(diff)

			if scalar_diff < self.stability_level:
				not_stable = False

			i = i+1


			current_state_return = [current_state[j] for j in range(self.N - self.n_to_return, self.N)]
			new_state_return = [new_state[j] for j in range(self.N - self.n_to_return, self.N)]
			return_diff = sum([(p-q)*(p-q) for p, q in zip(current_state_return, new_state_return)])


		if self.print_debug:
			print "Stable after ", i
			print "scalar_diff ", scalar_diff
			print "return_diff ", return_diff 



		
	def step(self, inputs):		
		if len(inputs) > self.N:
			print "len(inputs) > self.N"
			return
		
		for i in range(0, len(inputs)):
			self.neurons[i].value = inputs[i]

		self._stabilize()
		

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