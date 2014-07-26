import random
import math
import copy

MUTATION_RATE = 0.67
MUTATION_MEAN = 0
MUTATION_SIGMA = 0.05

STABILITY_LEVEL = 0.001
STABILITY_ITERATIONS_MAX = 500
STABILITY_ITERATIONS_MIN = 10

class WeightsDict:
	def __init__(self):
		self.weights = {}

	def __getitem__(self, key):
		key.sort()
		return self.weights[(key[0], key[1])]

	def __setitem__(self, key, value):
		key.sort()
		self.weights[(key[0], key[1])] = value

	def __str__(self):
		return str(self.weights)
class MLPLayer:
	def __init__(self, n_neurons, n_inputs):
		self.n_neurons = n_neurons
		self.n_inputs = n_inputs
		
		self._weights = []

		for i in range(self.n_neurons):
			self._weights.append([0] * (1 + self.n_inputs))

	def step(self, inputs):
		print inputs
		inputs.append(1)
		output = []
		for w in self._weights:
			inputs_weighted = sum([x*y for x,y in zip(w, inputs)])
			output.append(1/(1 + math.exp(-inputs_weighted)))

		return output

class MLPNetwork:
	def __init__(self, n_layers, n_neurons_per_layer):
		self._layers = []
		self._input_size = n_neurons_per_layer[0]

		for i in range(1, n_layers):
			self._layers.append(MLPLayer(n_neurons_per_layer[i], n_neurons_per_layer[i-1]))

	def step(self, inputs):
		if len(inputs) != self._input_size:
			print "Input size does not match."	
			return 

		last_result = inputs

		for l in self._layers:
			last_result = l.step(last_result)

		return last_result



class NeuroSystem:
	def __init__(self, N, n_to_returnzz):
		self.N = N
		self.n_to_return = n_to_return
		self.beta = 5
		
		self.print_debug = False

		self.weights = WeightsDict()
		self.current_state = [0 for i in range(N)]

		#wieghts between neurons
		for i in range (0, N):
			for j in range (i, N):
				self.weights[[i, j]] = random.uniform(-1, 1)

		#no self feedback
		for i in range(N):
			self.weights[[i, i]] = 0


		global STABILITY_LEVEL
		global STABILITY_ITERATIONS_MIN
		global STABILITY_ITERATIONS_MAX


		self.stability_level = STABILITY_LEVEL
		self.stability_iterations_min = STABILITY_ITERATIONS_MIN
		self.stability_iterations_max = STABILITY_ITERATIONS_MAX

	def __str__(self):
		res = "N: " + str(self.N) + " " + ', '.join([str(s) for s in self.current_state])
		if self.print_debug:
			res = res + "\n" + str(self.weights)

		return res

	def _advance(self):	
		update_id = random.randrange(0, self.N)

		#if self.print_debug:
		#	print "update_id: ", update_id
		#	print "current_state[update_id]: ", self.current_state[update_id]

		inputs = [self.weights[[i, update_id]] * self.current_state[i] for i in range (self.N)]

		#self.current_state[update_id] = 1/(1 + math.exp(-1 * self.beta * sum(inputs)))

		#if self.print_debug:
		#	print "inputs: ", sum(inputs), " for id: ", update_id

		new_state = 0
		if sum(inputs) > 0:
			new_state = 1
		else:
			new_state = -1

		if self.print_debug:
			if new_state != self.current_state[update_id]:
				print "Changed value of ", update_id, " from ", self.current_state[update_id], " to ", new_state

		self.current_state[update_id] = new_state


		#if self.print_debug:
		#	print "current_state[update_id]: ", self.current_state[update_id]
			


	def _proceed(self):
		i = 0
		not_stable = True
		scalar_diff = 0
		return_diff = 0

		while i < self.stability_iterations_min or (i < self.stability_iterations_max and not_stable):
			current_state = list(self.current_state)
			
			for j in range(self.N):
				self._advance()

			new_state = self.current_state

			diff = [(p-q)*(p-q) for p, q in zip(current_state, new_state)]
			scalar_diff = sum(diff)

			if scalar_diff < self.stability_level:
				not_stable = False
			else:
				not_stable = True

			i = i+1


			current_state_return = [current_state[j] for j in range(self.N - self.n_to_return, self.N)]
			new_state_return = [new_state[j] for j in range(self.N - self.n_to_return, self.N)]
			return_diff = sum([(p-q)*(p-q) for p, q in zip(current_state_return, new_state_return)])


		if self.print_debug:
			print "Stable after ", i
			#print "scalar_diff ", scalar_diff
			#print "return_diff ", return_diff 


		
	def step(self, inputs):		
		if len(inputs) > self.N:
			print "len(inputs) > self.N"
			return

		self.current_state = [0 for i in range(self.N)]
		
		for i in range(0, len(inputs)):
			self.current_state[i] = inputs[i]

		self._proceed()
		

		return [self.current_state[i] for i in range(self.N - self.n_to_return, self.N)]
		

		return ret

	def _cross(self, partner):
		if self.N != partner.N:
			print "We cannot DO it!"
			return

		new_weights = WeightsDict()

		for i in range(0, self.N):
			for j in range (i, self.N):
				toss = bool(random.getrandbits(1))
				if (toss):
					new_weights[[i, j]] = self.weights[[i, j]]
				else:
					new_weights[[i, j]] = partner.weights[[i, j]]

		offspring = NeuroSystem(self.N, self.n_to_return)
		offspring.weights = new_weights

		return offspring

	def _mutate(self):
		global MUTATION_RATE
		global MUTATION_MEAN
		global MUTATION_SIGMA

		for i in range (0, self.N):
			for j in range (i, self.N):
				rand = random.uniform(0, 1)
				if rand < MUTATION_RATE:
					change = random.normalvariate(MUTATION_MEAN, MUTATION_SIGMA)
					self.weights[[i, j]] = self.weights[[i, j]] + change

					if self.print_debug:
						print "Changed weights ", i, " ", j, " by ", change

		for i in range(self.N):
			self.weights[[i, i]] = 0		


	def reproduce(self, partner):
		offspring = self._cross(partner)
		offspring._mutate()

		return offspring
	


