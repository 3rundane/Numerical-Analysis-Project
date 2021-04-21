import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

## Returns rightmost endpoint of euler's method on the interval [0,1] with stepsize h.
def euler_m(h):
    y_current = 1
    n_steps = int(1/h)

    for i in range(0,n_steps):
        y_current = y_current + h*y_current
    # print(y_current)
    return y_current

## Returns rightmost endpoint of the trapezoidal rule on the interval [0,1] with stepsize h.
def trap_m(h):
    y_current = 1
    n_steps = int(1/h)
    for i in range(0,n_steps):
        y_current = ((1+(h/2))/(1-(h/2)))*y_current
    return y_current

## Returns rightmost endpoint of simpsons rule on the interval [0,1] with stepsize h.
def simpsons_m(h):
    y_current = 1 #n
    y_next = np.exp(h)  # n+1
    y_hold = 0
    n_steps = int(1/h)
    for i in range(0,n_steps-1):
        y_hold = y_next
        # y_next = ((1 + (h / 3)) / (1 - (h / 3)))*y_current + (4*h/3)*y_next/(1-h/3)#calculates n+2
        # print(((1 + (h / 3)) / (1 - (h / 3)))*y_current + (4*h/3)*y_next/(1-h/3) - ((3+h)*y_current + 4*h*y_next) / (3-h)) #calculates n+2)
        y_next = ((3+h)*y_current + 4*h*y_next) / (3-h)

        y_current = y_hold
    return y_next

## Returns rightmost endpoint of runge-kutta method on the interval [0,1] with stepsize h.
def rk4_m(h):
    y_current = 1
    k_1 = 0
    k_2 = 0
    k_3 = 0
    k_4 = 0
    n_steps = int(1/h)
    for i in range(0,n_steps):
        k_1 = y_current
        k_2 = y_current + (h/2)*k_1
        k_3 = y_current + (h/2)*k_2
        k_4 = y_current + h*k_3

        y_current = y_current + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    return y_current

def main():
    s = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])
    h = np.power(0.5,np.flip(s))
    # print(np.flip(s))
    right_endpoint = np.exp(1)
    euler_error = np.array([])
    trap_error = np.array([])
    simpsons_error = np.array([])
    rk4_error = np.array([])
    for step in h:

        euler_error = np.append(euler_error,np.abs(right_endpoint - euler_m(step)))
        trap_error = np.append(trap_error,np.abs(right_endpoint - trap_m(step)))
        simpsons_error = np.append(simpsons_error,np.abs(right_endpoint - simpsons_m(step)))
        rk4_error = np.append(rk4_error,np.abs(right_endpoint - rk4_m(step)))
    # print(euler_error)
    # fig,axes = plt.subplots(2,2)
    # axes[0][0].loglog(h,euler_error,'b.',label="euler error",base=2)
    # axes[0][0].set_title("euler error")
    #
    # axes[1][0].loglog(h,trap_error,'k.',label="trapezoidal error",base=2)
    # axes[1][0].set_title("trapezoidal error")
    #
    # axes[0][1].loglog(h,simpsons_error,'r.',label="simpsons error",base=2)
    # axes[0][1].set_title("simpsons error")
    #
    # axes[1][1].loglog(h,rk4_error,'m.',label="Runge-Kutta",base=2)
    # axes[1][1].set_title("rk4 error")

    ##compute slope qualitatively
    slope_euler = (np.log2(euler_error[-1]) - np.log2(euler_error[0]))/(np.log2(h[-1]) - np.log2(h[0]))
    slope_trap = (np.log2(trap_error[-1]) - np.log2(trap_error[0]))/(np.log2(h[-1]) - np.log2(h[0]))
    slope_simps = (np.log2(simpsons_error[8]) - np.log2(simpsons_error[6]))/(np.log2(h[8]) - np.log2(h[6]))
    slope_rk4 = (np.log2(rk4_error[8]) - np.log2(rk4_error[6]))/(np.log2(h[8]) - np.log2(h[6]))

    ## print out qualitative slopes
    print(slope_euler)
    print(slope_trap)
    print(slope_simps)
    print(slope_rk4)

    plt.loglog(h,euler_error,'cyan',label="euler error (1)",base=2)
    plt.loglog(h,trap_error,'k',label="trapezoidal error (2)",base=2)
    plt.loglog(h,simpsons_error,'r',label="simpsons error (4)",base=2)
    plt.loglog(h,rk4_error,'purple',label="Runge-Kutta (4)",base=2)
    # plt.semilogy(h, euler_error, 'orange', label="euler error (1)", base=2)
    # plt.semilogy(h, trap_error, 'k', label="trapezoidal error (2)", base=2)
    # plt.semilogy(h, simpsons_error, 'r', label="simpsons error (4)", base=2)
    # plt.semilogy(h, rk4_error, 'purple', label="Runge-Kutta (4)", base=2)
    # plt.plot(h, euler_error, 'b', label="euler error")
    # plt.plot(h, trap_error, 'k', label="trapezoidal error")
    # plt.plot(h, simpsons_error, 'r', label="simpsons error")
    # plt.plot(h, rk4_error, 'purple', label="Runge-Kutta")
    plt.ylabel("log_2 of error at the right end point")

    #Plot a rough estimate of the machine limit based on the points where the last two methods go awry.
    plt.axhline(y=7*(1/10)**(15), color='blue', linestyle='--',label="rough precision limit")
    plt.xlabel("log_2 of Step-size h")
    plt.legend()
    plt.title("Log-Log plot (base 2) of right endpoint error vs. step-size (h)")
    plt.show()




if __name__ == '__main__':
    main()