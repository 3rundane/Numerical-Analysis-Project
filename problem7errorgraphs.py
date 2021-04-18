import matplotlib.pyplot as plt
import numpy as np
import sys

our_epsilon = sys.float_info.epsilon

# Let x represent epsilon
def function(x):
    numerator = np.log(2 - 2*x + x**2)
    denominator = np.sqrt(2*x - x**2)
    output  = 2/3*(1-x)*numerator / denominator
    return output

def main():
    fudge_factor = our_epsilon
    domain = np.linspace(0+fudge_factor, 5*fudge_factor,50)
    plt.plot(domain, function(domain))
    plt.title("Simpons Rule - Average")
    plt.xlabel('Epsilon')
    plt.ylabel('Error')
    plt.show()

if __name__ == '__main__':
    main()
