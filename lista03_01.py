def is_prime(x):
	return not [y for y in range(2, x) if x % y == 0]
	
def prime_list_coh(n):
	print [x for x in range(2,n) if is_prime(x)]
	
def prime_list_fil(n):
	print filter(is_prime, range(2, n))

prime_list_coh(20)

prime_list_fil(20)




