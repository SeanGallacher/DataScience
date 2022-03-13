# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
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
from scipy import optimize
# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=200):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    x0 = (a+b)/2
    p = (1+ 5**.5)/2
    its = 0
    convergence = False
    # iterate only maxiter times at most.
    for i in range(maxiter):
        c = (b-a)/p
        at = b - c
        bt = a+c
        # Get new boundaries for the search interval.
        if f(at) <= f(bt):
            b = bt
        else:
            a = at
        # Set the minimizer approximation as the interval midpoint.
        x1 = (a+b)/2
        its +=1
        # Stop iterating if the approximation stops changing enough.
        if np.abs(x0-x1) < tol:
            convergence =  True
            break
        x0 = x1

    return x1,  convergence, its



# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    it = 0
    change = 1
    ## use the one d method to minimize a function
    while it < maxiter and change > tol:
        xk = x0 - df(x0)/d2f(x0)
        change = np.abs(xk-x0)
        it +=1
        x0 = xk

    return x0, change < tol, it


def o():
    f = lambda x: np.exp(x) - 4*x
    #print(golden_section(f,0,3))
    # print(optimize.golden(f,brack=(0,3), tol=.001))

    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)
    print(optimize.newton(f,0,df))
    print(newton1d(df, d2f, 0, maxiter=500))



# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
        """Use the secant method to minimize a function f:R->R.

        Parameters:
            df (function): The first derivative of f.
            x0 (float): An initial guess for the minimizer of f.
            x1 (float): Another guess for the minimizer of f.
            tol (float): The stopping tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            (float): The approximate minimizer of f.
            (bool): Whether or not the algorithm converged.
            (int): The number of iterations computed.
        """
        it = 0
        conv = False
        dfx0 = df(x0)
        ### Use difference Quocients to compute the derivative
        while it < maxiter:
            dfx1 = df(x1)
            xk = (x0*dfx1-x1*dfx0)/(dfx1-dfx0)

            if np.abs(xk-x1) < tol:
                conv = True
                break
            it +=1

            x0 = x1
            x1 = xk

            dfx0 = dfx1

        return xk, conv, it



def p3():
    f = lambda x: x ** 2 + np.sin(x) + np.sin(10 * x)
    df = lambda x: 2 * x + np.cos(x) + 10 * np.cos(10 * x)
    print(secant1d(df, 0, -1))




# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    ### Implement backtracking to find a great alpha
    Dfp = Df(x)@p
    fx = f(x)
    while f(x +alpha*p) > fx + c*alpha*Dfp:
        alpha = rho*alpha
    return alpha


if __name__ == '__main__':
    f = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    Df = lambda x: np.array([2 * x[0], 2 * x[1], 2 * x[2]])
    x = np.array([150,.03,40])
    #print(backtracking(f, Df,x, np.array([-.5,-100,-.45]) ))


    o()

    if True:

        f = lambda x: x**2 + np.sin(5*x)
        df = lambda x: 2*x + 5*np.cos(5*x)
        df2 = lambda x: 2 - 25*np.sin(5*x)
        a = -2.5
        b = 5
        xlin = np.linspace(a,b,100)

        plt.plot(xlin,f(xlin))
        x1 = golden_section(f,a,b)[0]
        plt.scatter(x1, (f(x1)))

        plt.show()

        ## test 2
        plt.plot(xlin,f(xlin))
        x2 = newton1d(df,df2, 0, tol=1e-10,maxiter=50)[0]
        plt.scatter(x2, (f(x2)))
        plt.show()


        ### test 3
        f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
        df = lambda x: 2 * x + np.cos(x) + 10 * np.cos(10 * x)

        xlin = np.linspace(-1,3,100)
        plt.plot(xlin, f(xlin) )
        x3 = secant1d(df, 0,-1, tol=1e-10, maxiter=500)[0]
        plt.scatter(x3, (f(x3)))

        plt.show()


if __name__ == '__main__':
    if False:
        from scipy import optimize as opt
        import numpy as np
        f = lambda x: np.exp(x) - 4 * x

        print(opt.golden(f, brack=(0, 3), tol=.00000000001))
        print(golden_section(f, 0, 3,maxiter=100))
        print()
        print(golden_section(lambda x: x ** 2 - 1, -3, 4, tol=1e-12, maxiter=40))
        print(opt.golden(lambda x: x ** 2 - 1, brack=(0, 10), tol=.00000000001))
        print()
        print(golden_section(lambda x: np.exp(x) - 4 * x, 0, 3, tol=1e-12, maxiter=500))
        print(opt.golden(lambda x: np.exp(x) - 4 * x, brack=(0, 10), tol=.00000000001))

        print(golden_section(lambda x: x ** 2 - 1, -3, 4, tol=1e-12, maxiter=30))
        print(opt.golden(lambda x: x ** 2 - 1, brack=(0, 10), tol=.00000000001))


