import matplotlib.pyplot as plt
import numpy as np
from math import factorial

# I don't really know how to make this nice. I'm to lazy, will probably just use desmos.
t = np.linspace(-10, 10, 1000)

# I need this one for the ln function since I don't know how to deal with singularity of -1
t_ln = np.linspace(-0.9999, 10, 1000)

# Define e^x
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

for n in range(1,21):
    plt.plot(t, f(t), color='blue', label='e^x')
    plt.plot(t, p(t, n), color='blue', linestyle='--', label='p')
    plt.plot(t_ln, g(t_ln), color='red', label='ln(1+x)')
    plt.plot(t, q(t, n), color='red', linestyle='--', label='q')
    plt.ylim(-10, 10)
    # plt.axvline(x=1)
    plt.title(f'Degree {n} Taylor Polynomial')
    plt.legend(loc=1)
    plt.show()
