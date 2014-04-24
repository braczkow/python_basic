import timeit

iterations = 10000

def is_prime(x):
	return not [y for y in range(2, x) if x % y == 0]
	
def prime_list_coh(n):
	return [x for x in range(2,n) if is_prime(x)]
	
def prime_list_fil(n):
	return filter(is_prime, range(2, n))

print timeit.timeit('prime_list_coh(100)', 
'from __main__ import prime_list_coh', number = iterations)

print timeit.timeit('prime_list_fil(100)', 
'from __main__ import prime_list_fil', number = iterations)



