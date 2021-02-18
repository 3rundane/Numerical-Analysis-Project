import numpy as np
from math import factorial
import matplotlib.pyplot as plt

def f(x):
    return np.exp(x)

# Define ln(x+1)
def g(x):
    return np.log(x+1)

# Define taylor poly for e^x
def p(x,n):
    if n == 0:
        return 1
    else:
        return ((x**n) / factorial(n)) + p(x, n-1)

# Define taylor poly for ln(x+1)
def q(x,n):
    if n == 1:
        return x
    else:
        return (-1)**(n-1) * factorial(n-1)*x**n / factorial(n) + q(x, n-1)

def error_f(x,n):
     return abs(f(x) - p(x, n))

def error_g(x,n):
    if x <= -1:
        return 'undefined'
    else:
        return abs(g(x) - q(x, n))

# x = float(input("x value: \n"))
# n = int(input("n range: \n"))

x = 1.5
n = 30


domain = [i for i in range(1, n+1)]
codomain_f = [error_f(x, i) for i in range(1, n+1)]
codomain_g = [error_g(x, i) for i in range(1, n+1)]

print("e^x errors:")
print('----------')
for i in range(0,n):
    print(f'degree {i+1}: {codomain_f[i]}')
print()

print("ln errors:")
print('----------')
for i in range(0,n):
    print(f'degree {i+1}: {codomain_g[i]}')

# plotting error
x_ticks = np.arange(1, n+1, 1)
plt.xticks(x_ticks)
plt.title(f'Error at x={x}')
plt.scatter(x=domain, y=codomain_f, color='blue', label='e^x error')
plt.scatter(x=domain, y=codomain_g, color='red', label='ln error')
plt.xlabel('degree n')
plt.ylabel('error')
plt.legend()
plt.show()