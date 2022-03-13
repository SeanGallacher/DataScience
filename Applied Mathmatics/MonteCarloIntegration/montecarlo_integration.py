# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""

import numpy as np
from numba import jit
from scipy import stats
from matplotlib import pyplot as plt
import numba

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """

    # V = np.pi**(n/2) / gamma(n/2 + 1) actual volume
    NBox = 2**n
    sample = np.array([np.random.uniform(-1, 1, int(N/n)+1) for x in range(n)])

    # print(sample)
    lengths = np.sum(sample ** 2, axis=0) ** .5
    ballList = []
    for s in lengths:
        if s <= 1:
            ballList.append(1)
        else:
            ballList.append(0)

    ballAprox = NBox * sum(ballList) / (N/n)

    return ballAprox



# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    points = np.random.uniform(a,b,N)
    return np.sum(f(points)) * (b-a) /N


def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    n = len(maxs)
    v = np.prod([maxs[i]-mins[i] for i in range(n)])
    points = np.array([np.random.uniform(mins[i], maxs[i], N) for i in range(n)])
    fx = [f(point) for point in points.T]
    return (v/N)*np.sum(fx)





# Problem 3


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    n = 4
    f = lambda x: (1 / ((2*np.pi)**(n/2))) * np.exp(-x.T @ x * .5)
    # Define the bounds of integration.
    mins = np.array([-3/2, 0,0,0])
    maxs = np.array([3/4,1,.5,1])
    means, cov = np.zeros(n), np.eye(n)
    # Compute the integral with SciPy.
    exact = stats.mvn.mvnun(mins, maxs, means, cov)[0]

    NS = np.logspace(1,5,20)
    errors = []
    for N in NS:
        aprox = mc_integrate(f, mins, maxs, int(N))
        errors.append(np.abs(aprox-exact)/exact)

    plt.loglog(NS, errors, marker='.')
    plt.loglog(NS, 1/np.sqrt(NS), marker='.')
    plt.title('relative erorrs of monte carlo')
    plt.show()

import time
if __name__ == '__main__':
    #print(ball_volume(2, 100000))
    print(mc_integrate1d(lambda x: x**2, -4,2))
    print(mc_integrate1d(lambda x: np.sin(x), -2*np.pi, 2*np.pi))

    if True:
        f1 = lambda x: x[0]**2 + x[1]**2
        print("should be 2/3:", mc_integrate(f1, [0,0], [1,1]))
        f2 = lambda x: 3*x[0] - 4*x[1] + x[1]**2
        print("should be 54:", mc_integrate(f2, [1,-2], [3,1]))
        f3 = lambda x: x[0] + x[1] - x[3]*x[2]**2
        #result = np.mean([mc_integrate(f3, [-1,-2,-3,-4], [1,2,3,4],10000) for _ in range(100)])
        #print("should be 0:", result)

    st = time.time()
    print(prob4())
    print(time.time()-st)