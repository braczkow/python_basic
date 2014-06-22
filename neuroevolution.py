import random

random.seed(3)

class Neuron:
	def __init__(self, N, id):
		self.weights = []
		for i in range (0, N):
			self.weights.append(random.uniform(-1, 1));
		
		self.weights[id] = 0
		
	def __str__(self):
		return "(" + '; '.join([str(w) for w in self.weights]) + ")"

class NeuroSystem:
	def __init__(self, N):
		self.N = N;
		self.neurons = []
		for i in range(0, N):
			self.neurons.append(Neuron(N, i));
			
	def __str__(self):
		return "N: " + str(self.N) +'\n' + ', '.join([str(n) for n in self.neurons])
		#print "N: ", self.N
		#print str(self.neurons)
		
		
ns = NeuroSystem(3);

print(ns);
			
	