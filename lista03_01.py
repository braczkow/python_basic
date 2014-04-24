def is_prime(x):
	return not [y for y in range(2, x) if x % y == 0]
	
def prime_list_coh(n):
	print [x for x in range(2,n) if is_prime(x)]

prime_list_coh(20)




