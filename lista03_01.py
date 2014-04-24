def is_prime(x):
	return not [y for y in range(2, x) if x % y == 0]
	
print is_prime(8)

print is_prime(7)


