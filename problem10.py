#Chebycheff Polynomial
import matplotlib.pyplot as plt
import numpy as np
# from scipy import interpolate, linalg

# Define a linespace (or domain)
domain = np.linspace(-5, 5, 1000)

# Define the function
def f(x):
    return 1/(1+x**2)

# Interpolate with different degree polynomials
for n in range(1, 21):

    # creating our nodes
    x = [5*np.cos(i*np.pi/n) for i in range(n+1)]
    y = [f(i) for i in x]

    # Creating the Vondermond Matrix
    rows = []
    for i in x:
        row = [i**j for j in range(n+1)]
        rows.append(row)

    # Create my system that needs to be solved
    x_vondermond = np.array(rows)
    y_vondermond = np.array(y)

    # Solve the system of equations!
    sol = np.linalg.solve(x_vondermond, y_vondermond)

    # am dumb and didn't know how to build function. So I did it recursively.
    def poly(x, n):
        if n == 0:
            return sol[0]
        else:
            return sol[n]*x**n + poly(x, n-1)

    # Find the max error
    error = [abs(f(i)-poly(i, n)) for i in domain]
    max_error = max(error)
    max_error_index = error.index(max_error)
    print(f"Degree {n} Max Error: {max_error}")
    plot_value = -5 + max_error_index/100     # You divde by 1000 and then multiply by 10--think about it


    # Plot the results
    plt.plot(domain, f(domain), label='function', color='blue')
    plt.plot(domain, poly(domain,n), label='Chebycheff Interpolant')
    plt.text(0, 0.1, f'Max Error: {max_error}', horizontalalignment='center', verticalalignment='center')
    plt.vlines(x=plot_value, ymin=min(f(plot_value), poly(plot_value, n)), ymax=max(f(plot_value), poly(plot_value, n)), linestyle='--', color='red', label="Max Error")
    plt.title(f"Degree {n} Chebycheff Interpolation")
    plt.legend(loc=1)
    plt.show()


