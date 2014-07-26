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

class MLPNetworkEvolutor:
	def __init__(self, lhs, rhs):
		self._lhs = lhs
		self._rhs = rhs
		self._offspring = None

	
	def reproduce(self):
		_cross()
		_mutate()
	
	def _cross(self):





nn = MLPNetwork(3, [2, 2, 1])
nn.step([2, 3])
