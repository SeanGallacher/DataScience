# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt
import time
# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    ## return the derivative of f using sympy
    x = sy.symbols('x')
    f =  (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    return sy.lambdify(x, sy.diff(f, x))

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x+h) - f(x))/h
def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x + h) - f(x + 2*h)) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x-h) )/h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return  (f(x+h) - f(x-h))/(2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-2*h) - 8 * f(x - h) + 8* f(x + h) - f(x +2*h)) / (12 * h)


def prob2():
    ###plot the derivitives
    xlin = np.linspace(np.pi*-1, np.pi, 124)
    f = lambda x: (sy.sin(x) + 1) ** (sy.sin(sy.cos(x)))
    plt.plot(xlin, prob1()(xlin))
    plt.plot(xlin, [fdq1(f,x) for x in xlin])
    plt.plot(xlin, [fdq2(f,x)for x in xlin])
    plt.plot(xlin, [bdq1(f,x)for x in xlin])
    plt.plot(xlin, [bdq2(f,x)for x in xlin])
    plt.plot(xlin, [cdq2(f,x)for x in xlin])
    plt.plot(xlin, [cdq4(f,x)for x in xlin])
    plt.plot()
    plt.xlabel('x value')
    plt.ylabel('derivative value')
    plt.show()


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    ###initalize functions
    df = prob1()
    xlin = np.logspace(-8,0,9)
    f = lambda x: (anp.sin(x) + 1) ** anp.sin(anp.cos(x))

    ##plot the derivitives
    plt.loglog(xlin, [np.abs(fdq1(f,x0,x) -df(x0)) for x in xlin], label ='fdq1', marker = '.' )
    plt.loglog(xlin, [np.abs(fdq2(f,x0,x)-df(x0)) for x in xlin], label ='fdq2', marker = '.' )
    plt.loglog(xlin, [np.abs(bdq1(f,x0,x)-df(x0)) for x in xlin], label ='bdq1', marker = '.' )
    plt.loglog(xlin, [np.abs(bdq2(f,x0,x)-df(x0)) for x in xlin], label ='bdq2', marker = '.' )
    plt.loglog(xlin, [np.abs(cdq2(f,x0,x)-df(x0)) for x in xlin], label ='cdq2', marker = '.' )
    plt.loglog(xlin, [np.abs(cdq4(f,x0,x)-df(x0)) for x in xlin], label ='cdq4', marker = '.' )
    plt.legend()
    plt.xlabel('x value')
    plt.ylabel('Absolute error')
    plt.show()



# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    ##initialize data
    data = np.deg2rad( np.load('plane.npy')[:,1:])

    times = range(7,15)
    xf =  lambda a: 500*np.tan(a[1]) / (np.tan(a[1])-np.tan(a[0]))
    yf = lambda a: 500 * np.tan(a[1]) * np.tan(a[0]) / (np.tan(a[1]) - np.tan(a[0]))

    xp = []
    yp = []
    ###take the derivitives
    for i,t in enumerate(times):

        if t == 7:
            xp.append(xf(data[i+1])- xf(data[i]))
            yp.append(yf(data[i+1])- yf(data[i]))
        elif t==14:
            xp.append(xf(data[i]) - xf(data[i-1]))
            yp.append(yf(data[i]) - yf(data[i-1]))
        else:
            xp.append((xf(data[i+1]) - xf(data[i-1]))/2)
            yp.append((yf(data[i+1]) - yf(data[i-1]))/2)


    ### get the velocity
    v = np.array([np.sqrt(xp[i]**2 + yp[i]**2) for i in range(len(xp))])
    if max(v-np.array([46.42420062, 47.00103938, 48.99880514, 50.09944163, 48.29035084,
       51.56455905, 53.92303355, 51.51480057])) < .1:
        pass

    return v


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    ## get matrix dim
    N = len(x)
    M = len(f(x))
    J = (np.zeros((M,N))).T

    ### calculate the Jacobian
    for i in range(N):
        e = np.eye(N)[:,i]
        dfi = (f(x+h*e) - f(x-h*e))/(2*h)
        J[i, :] = dfi
    return J.T



from autograd import numpy as anp # Use autograd's version of NumPy.
from autograd import grad
from autograd import elementwise_grad

# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    ### get the chevychev polynomial
    return anp.cos(n * anp.arccos(x))

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    ### use Auto grad and plot graph
    xlin = np.linspace(-1,1,123)
    for k in range(5):
        f = grad(lambda x: cheb_poly(x,k))
        plt.plot(xlin, [f(x) for x in xlin], label='k = ' + str(k))
    plt.title("Grad for chebyshev polynomials")
    plt.show()


import random
# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    ##initialize lists
    f = lambda x: (anp.sin(x) + 1)**anp.sin(anp.cos(x))
    symTime = []
    symError = []
    difQuoTime = []
    difQuoError = []
    gradError = []
    gradTime = []

    for n in range(N):
        x0 = random.uniform(0, anp.pi*2)

        ###Sympy time
        s = time.time()
        df = prob1()
        exact = df(x0)
        e = time.time() - s

        symTime.append(e)
        symError.append(1e-18)


        ##Difference quotient time and error
        s = time.time()
        dqAprox = cdq4(f, x0)
        e = time.time() - s

        difQuoTime.append(e)
        difQuoError.append(np.abs(dqAprox-exact))

        ##Grad time and error
        s = time.time()
        df = grad(f)
        gradAprox = df(x0)
        e = time.time() - s

        gradTime.append(e)
        gradError.append(np.abs(gradAprox-exact))

    ##Plot graphs
    plt.scatter(symTime, symError, color='blue', label='SymPy', marker='o')
    plt.scatter(difQuoTime, difQuoError, color='purple', label='Difference Quotients', marker='o')
    plt.scatter(gradTime, gradError, color='red', label='Autograd', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('time')
    plt.ylabel('error')
    plt.title('time vs error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    f = lambda x: np.array([x[0]**2, x[0] ** 3 - x[1]])
    #f = lambda x: np.array([x[0]*x[1], x[1]*x[2], x[0] - x[1], 2 + x[2] ** 2])
    #print(jacobian_cdq2(f,(1,1)))
    #print(prob6())
    #prob2()
    #prob3(1)
    #print(prob4())
    #prob7()
    #print(1/(+0))