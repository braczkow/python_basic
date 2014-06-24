import random
import math

random.seed(3)

MUTATION_RATE = 0.05
MUTATION_MEAN = 0
MUTATION_SIGMA = 1

class Neuron:
	def __init__(self, N, id):
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

	def catchInputs(self, currentState):
		self.catchedInput = sum(p*q for p,q in zip(self.weights, currentState))

	def update(self):
		self.value = 1/(1 + math.exp(-1 * self.beta * self.catchedInput))

	def mutate(self):
		global MUTATION_RATE
		global MUTATION_MEAN
		global MUTATION_SIGMA

		for i in range (0, N):
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
		currentState = [n.value for n in self.neurons];
			
		for i in range(0, self.N):
			self.neurons[i].catchInputs(currentState);

		for i in range(0, self.N):
			self.neurons[i].update();
		
	def step(self, inputs):
		print "inputs: " + ", ".join([str(i) for i in inputs])
			
		if len(inputs) > self.N:
			print "len(inputs) > self.N"
			return
		
		for i in range(0, len(inputs)):
			self.neurons[i].value = inputs[i]

		#currentState = [n.value for n in self.neurons];
		#print currentState	

		#TODO add stabilization
		self._advance()

		#currentState = [n.value for n in self.neurons];
		#print currentState	

		return [self.neurons[i] for i in range(self.N - self.n_to_return, self.N)]

	def _cross(self, partner):
		if self.N != partner.N:
			print "We cannot DO IT!"
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

	def _one_generation(self):
		fitnesses = [p.fitness = fitness_function(p) for p in self.currentPopulation]
		sorted(self.currentPopulation, key=lambda p: p.fitness)

	def run(self, n_generations, n_population):
		self.n_generations = n_generations
		self.n_population = n_population

		self.currentPopulation = [class_type() for i in range(n_population)]

		for i in range(0, self.n_generations):
			self._one_generation()

		return self.leader


def NeuroSystemDef():
	return NeuroSystem(5, 1)

def xor_fitness(neuro_system):
	return 1


ga = GeneticAlgorithm(NeuroSystemDef, xor_fitness)


			
#ns = NeuroSystem(3, 1);
#print "Just created:"
#print "\n"
#print(ns);		

#result = ns.step([1, 2]);

#print "\nAfter step:"
#print(ns);	

#print "\nResult:"
#print [str(x) for x in result]


#ns_partner = NeuroSystem(3, 1)
#print "ns_partner " + str(ns_partner)


#child = ns.evolve(ns_partner)
#print "child " + str(child)

