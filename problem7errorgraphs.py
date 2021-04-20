import matplotlib.pyplot as plt
import numpy as np
import sys

our_epsilon = sys.float_info.epsilon
true_value = 2*np.pi*np.log((1+np.sqrt(2))/ 2)

# Let x represent epsilon
# Average area according to Simpson's rule
def function(x):
    numerator = np.log(2 - 2*x + x**2)
    denominator = np.sqrt(2*x - 2*x**2)
    output  = 2/3*(1-x)*numerator / denominator
    return output

def main():
    fudge_factor = our_epsilon
    domain = np.linspace(0+fudge_factor, 5*fudge_factor,50)
    plt.plot(domain, abs(true_value-function(domain)))
    plt.title("Simpons Rule - Average")
    plt.xlabel('Epsilon')
    plt.ylabel('Error')
    plt.show()

if __name__ == '__main__':
    main()
