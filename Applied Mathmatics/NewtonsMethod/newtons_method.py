# newtons_method.py
"""Volume 1: Newton's Method.
<Name>
<Class>
<Date>
"""
import sympy as sy
import numpy as np
from matplotlib import pyplot as plt
import time
from autograd import numpy as anp # Use autograd's version of NumPy.
from autograd import grad
from autograd import elementwise_grad
from autograd import jacobian
from scipy import linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    ## inti varables
    it = 0
    change = 1
    ### run until stop codition is hit
    while it < maxiter and change > tol:
        ### run newtowns method differently for scalars and vectors
        if not np.isscalar(x0):
            yk = la.solve(Df(x0),f(x0))
        else:
            yk = f(x0)/Df(x0)
        xk = x0 - alpha*yk

        if np.isscalar(xk):
            change = np.abs(xk-x0)
        else:
            change = la.norm(xk-x0)#np.max([xk[i]-x0[i] for i in range(len(xk))])
        x0 = xk

        it+=1

    return x0, change < tol, it





# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    ### run newtons method to predict returns
    f = lambda r: (P2*(1-(1+r)**-N2))/(P1*((1+r)**N1 - 1)) - 1
    df = grad(f)
    return newton(f,.05, df, maxiter=5000)[0]



# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    lin = np.linspace(1e-1,1, 100)
    its = []
    ### plot newtons for variable alphas
    for a in lin:
        x, bol, it = newton(f, x0, Df, alpha=a)
        its.append(it)
    ### return the best alpha
    minaIndex = its.index(min(its))
    plt.plot(lin, its)
    plt.show()
    return lin[minaIndex]


#import jaxlib
#from jax import jacfwd, jacrev
# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    ### intialize Equtions
    h1 =  lambda x:  5*x[0]*x[1] - x[0]*(1 + x[1])
    h2 = lambda x:  -x[0]*x[1] + (1 - x[1])*(1 + x[1])
    f = lambda x:  (5*x[0]*x[1] - x[0]*(1 + x[1]), -x[0]*x[1] + (1 - x[1])*(1 + x[1]))
    df = lambda x: np.array([[5.*x[1] - (1.+ x[1]), 5*x[0] - x[0]],
                             [-x[1], -x[0]-2*x[1]]])

    ### set up lin spaces
    k = 200
    xlin = np.linspace(-.1, -.001, k)
    ylin = np.linspace(.2, .25, k)
    ### look for changing alphas
    for x,y in zip(xlin,ylin):
        v1 = newton(f, np.array([x,y]), df, alpha=1, maxiter=500)[0]
        v2 = newton(f, np.array([x,y]), df, alpha=.55, maxiter=500)[0]
        ## if the correct alphas are found, return
        if (la.norm(v1 - np.array([0,1]) )< 1e-4 or la.norm(v1 - np.array([0,-1]) )< 1e-4) and la.norm(v2 - np.array([3.75,.25]) ) < 1e-4:
            return np.array([x,y])

    return 'nothing'




# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    ### set up lin spaces
    x_real = np.linspace(domain[0], domain[1], res)  # Real parts.
    x_imag = np.linspace(domain[2], domain[3], res)  # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j * X_imag

    for i in range(iters):
        X_1 = X_0 - f(X_0) / Df(X_0)
        X_0 = X_1
    ### find the closest point of convergence
    def closest(i,j):
        get_closest = lambda list_value: abs(list_value - X_1[i][j])
        return list(zeros).index(min(zeros, key=get_closest))
    Y = np.array([[closest(j,i) for i in range(len(X_0[0]))] for j in range(len(X_0[1]))])

    ## plot Graphs
    plt.pcolormesh(x_real,x_imag,Y, cmap="brg")
    plt.show()



if __name__ == '__main__':
    if True:
        print('testing newton')
        f = lambda x: np.sign(x) * np.power(np.abs(x), 1. / 3)
        f = lambda x: x**2 -1
        df = grad(f)
        print(newton(f,.01,df,alpha=1))

        print('testing alpha optimal')

        print(optimal_alpha(f, .01, df, tol=1e-18))



        f = lambda x: x**3 - x

        df = lambda x: 3*x**2 -1

        plot_basins(f, df, np.array([1, -.5 + (3**.5j/2), -.5 - (3**.5j/2)]), [-1.5,1.5,-1.5,1.5])

        prob6()
        #prob2(1,1,12,14)

        f = lambda x, z: x**2 + z**3+ z
