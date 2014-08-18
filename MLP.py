import random
import math

MUTATION_MEAN = 0
MUTATION_SIGMA = 0.5

class MLPLayer:
	def __init__(self, n_neurons, n_inputs):
		self.n_neurons = n_neurons
		self.n_inputs = n_inputs
		
		self._weights = []

		for i in range(self.n_neurons):
			self._weights.append([random.normalvariate(0, 1)] * (1 + self.n_inputs))

		self.print_debug = False


	def __str__(self):
		ret = ""
		i = 0
		for n in self._weights:
			ret = ret + "#" + str(i) + " "
			i = i+1 
			ret = ret + str(n)
			ret = ret + "\n"

		return ret


	def step(self, inputs):
		if self.print_debug == True:
			print "Layer input: " + str(inputs)
	
		inputs.append(1)
		output = []
		for w in self._weights:
			inputs_weighted = sum([x*y for x,y in zip(w, inputs)])
			output.append(2/(1 + math.exp(-inputs_weighted)) -1 )

		return output


	

class MLPNetwork:
	def __init__(self, n_layers, n_neurons_per_layer):
		self._layers = []
		self._input_size = n_neurons_per_layer[0]

		self.n_layers = n_layers
		self.n_neurons_per_layer = n_neurons_per_layer

		for i in range(1, n_layers):
			self._layers.append(MLPLayer(n_neurons_per_layer[i], n_neurons_per_layer[i-1]))

		self.print_debug = False

	def __str__(self):
		ret = ""
		i = 0
		for l in self._layers:
			ret = ret + "###" + str(i) + "\n"
			i = i+1
			ret = ret + str(l)
		
		return ret

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
		self.print_debug = False

	
	def reproduce(self):
		self._cross()
		
		if self.print_debug == True:
			print "MLNetworkEvolutor : after _cross"
			print self._offspring

		self._mutate()
		
		if self.print_debug == True:
			print "MLNetworkEvolutor : after_mutate"
			print self._offspring

		return self._offspring
		
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

	def _mutate_neuron(self, neuron):
		if self.print_debug == True:
			print "_mutate_neuron : " + str(neuron)

                global MUTATION_MEAN
                global MUTATION_SIGMA
                
		ret =  [(w+random.normalvariate(MUTATION_MEAN, MUTATION_SIGMA)) for w in neuron]
		
		if self.print_debug == True:
			print "_nutate_neuron : ret = " + str(ret)
		return ret

	def _mutate_layer(self, layer):
		if self.print_debug == True:
			print "_mutate_layer : \n" + str(layer)
		
		new_layer = MLPLayer(layer.n_neurons, layer.n_inputs)		
		ret = [self._mutate_neuron(n) for n in layer._weights]
		new_layer._weights = ret
		
		if self.print_debug == True:	
			print "_mutate_layer : ret=\n" + str(new_layer)
	
		return new_layer

	def _mutate(self):
		self._offspring._layers = [self._mutate_layer(l) for l in self._offspring._layers]

	def get_offspring(self):
		return self._offspring

class MLP_NE_System:
	def __init__(self):
		self._network = MLPNetwork



if __name__ == "__main__":

	print "MLPLayer: "
	layer = MLPLayer(3, 2)
	
	print str(layer)
	
	print "== MLPNetwor"
	nn = MLPNetwork(3, [2, 2, 1])
	
	print "stepping: "
	print nn.step([2, 3])
	
	print "== print MLPNetwork"
	print str(nn)
	
	nn2 = MLPNetwork(3, [2, 2, 1])
	
	mne = MLPNetworkEvolutor(nn, nn2)
	mne.reproduce()
	
	offspring = mne.get_offspring()
	
	print "== offspring:"
	print str(offspring)

