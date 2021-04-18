import numpy as np

# Preamble

# the true value of integral
true_value = 2*np.pi*np.log((1+np.sqrt(2))/ 2)

# Define our functions
def numerator(x):
    value = np.log(x**2 + 1)
    return value

# Method for computing integral
def quadrature(n):
    almost_f_value = 0
    for i in range(1,n+1):
        input = np.cos((2*i - 1)*np.pi / (2*n))
        almost_f_value = almost_f_value + numerator(input)
    f_value = np.pi / n * almost_f_value
    return f_value


# Main portion of program
def main():
    n_values = [4,5,6,7,8]
    for n in n_values:
        quad_value = quadrature(n)
        error = abs(true_value - quad_value)
        print(f"n value: {n}\nError: {error}\n")

if __name__ == '__main__':
    main()
