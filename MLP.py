import random
import math


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

		self.n_layers = n_layers
		self.n_neurons_per_layer = n_neurons_per_layer

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

class MLPNetworkEvolutor:
	def __init__(self, lhs, rhs):
		self._lhs = lhs
		self._rhs = rhs
		self._offspring = None

	
	def reproduce(self):
		self._cross()
		self._mutate()
		
	def _cross(self):
		if len(self._lhs._layers) != len(self._rhs._layers):
			print "MLPNetworkEvolutor._cross we are not of same size"
			return

		self._offspring = MLPNetwork(self._lhs.n_layers, self._lhs.n_neurons_per_layer)
		
		offspring_layers = []

		for lhs_layer, rhs_layer in zip(self._lhs._layers, self._rhs._layers):
			layer = self._cross_layers(lhs_layer, rhs_layer)
			offspring_layers.append(layer)

		self._offspring._layers = offspring_layers

	def _cross_layers(self, lhs, rhs):
			layer = MLPLayer(lhs.n_neurons, lhs.n_inputs)

			weights = []

			for lhs_weight, rhs_weight in zip(lhs._weights, rhs._weights):
				rand = random.uniform(0, 1)
				if rand > 0.5:
					weights.append(lhs_weight)
				else:
					weights.append(rhs_weight)

			layer._weights = weights

			return layer

			

	def _mutate(self):
		for l in self._offspring._layers:
			for n in l._weights:
				for w in n:
					w = w + random.normalvariate(0, 0.1)

	def get_offspring(self):
		return self._offspring

class MLP_NE_System:
	def __init__(self):
		self._network = MLPNetwork




nn = MLPNetwork(3, [2, 2, 1])
nn.step([2, 3])

nn2 = MLPNetwork(3, [2, 2, 1])

mne = MLPNetworkEvolutor(nn, nn2)
mne.reproduce()

offspring = mne.get_offspring()



