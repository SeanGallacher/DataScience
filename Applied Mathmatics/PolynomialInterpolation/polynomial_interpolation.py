# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    n = len(xint)

    ### array of demoninators for Lj (this is the same for
    L_demon = np.array([np.prod([xint[j] - xint[k] for k in np.delete(np.array(range(n)), j)]) for j in range(n)])

    ### array of demoninators for Lj each inner list is the m values for the Lx
    L_num = np.array([[np.prod([x - xint[k] for k in np.delete(np.array(range(n)), j)]) for x in points] for j in range(n)])
    L = L_num.T/L_demon

    p = np.sum(yint*L, axis =1)
    return p

def prob2():
    ##Create PLot using problem 1
    x = np.linspace(-1,1, 11)
    x2 = np.linspace(-1, 1, 100)
    f = lambda x: 1/(1+(25*x**2))

    p = lagrange(x, f(x), x2)
    plt.plot(x2, p)

    plt.plot(x2,f(x2))
    plt.show()


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        ## initialize varables
        self.X = xint
        self.Y = yint
        self.n = len(xint)
        ### Calculate weights
        self.Wjs = np.array([1/np.prod([xint[j] - xint[k] for k in np.delete(np.array(range(self.n)), j)]) for j in range(self.n)])

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        ### calcuate points for the polynomial
        h1 = lambda x: np.sum([self.Wjs[j]*self.Y[j] / (x-self.X[j]) for j in range(self.n)])
        h2 = lambda x: np.sum([self.Wjs[j]/(x-self.X[j]) for j in range(self.n)])
        p = lambda x:  h1(x)/h2(x)

        return np.array([p(x) for x in points])
    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        ##Append Y values
        self.Y = np.append( self.Y, yint)
        w = lambda x: 1/np.product([x - xj for xj in self.X])
        for i in range(len(xint)):
            ### update current weights
            self.Wjs = np.array([self.Wjs[j] / (self.X[j] - xint[i]) for j in range(len(self.Wjs))])
            ##append new weights
            self.Wjs = np.append(self.Wjs, w(xint[i]))
            self.X = np.append(self.X, xint[i])


        self.n = len(self.X)


def prob3():
    ### Use Barycentric to plot our function
    x = np.linspace(-1,1, 11)
    x2 = np.linspace(-1, 1, 100)
    f = lambda x: 1/(1+(25*x**2))
    baram = Barycentric(x,f(x))
    p = baram.__call__(x2)

    plt.plot(x2, p)

    plt.plot(x2,f(x2))
    plt.show()


def teEST4():
    ### test Adding Weights
    n = 11
    runge = lambda x: 1 / (1 + 25 * x ** 2)
    xvals_original = np.linspace(-1, 1, n)
    xvals_1 = xvals_original[1::2]
    xvals_2 = xvals_original[::2]
    domain = np.linspace(-1, 1, 1000)
    bary = Barycentric(xvals_1, runge(xvals_1))

    bary_2 = Barycentric(xvals_original, runge(xvals_original))
    plt.plot(domain, bary_2(domain), linewidth=6, label='Not added')
    plt.plot(domain, runge(domain), label='Original')
    plt.plot(domain, bary(domain), label='Odd Points, n = ' + str(n))
    bary.add_weights(xvals_2, runge(xvals_2))
    plt.plot(domain, bary(domain), 'k', label='All points, n = ' + str(n))
    plt.legend(loc='best')
    plt.show()


from scipy import linalg as la
from scipy.interpolate import BarycentricInterpolator

# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    ### initilize variables
    f = lambda x: 1 / (1 + (25 * x ** 2))
    errorsEven = []
    errorsRunge = []
    a = -1
    b = 1
    r = lambda x: .5 *(a+b + (b-a)*np.cos(x*np.pi/n))

    ###
    numPoints = [2**k for k in range(2,9)]
    xlin = np.linspace(-1,1,400)

    ## calculate the errors based on number of points
    for n in numPoints:
        ##Caculate interpolating points
        ptsEven = np.linspace(-1,1, n)
        ptsRunge = np.cos(np.pi*np.arange(0,n+1)/n)
        #ptsRunge = np.array([r(x) for x in ptsEven])

        poly1 = BarycentricInterpolator(ptsEven, f(ptsEven))
        poly2 = BarycentricInterpolator(ptsRunge, f(ptsRunge))

        ##append errors to list
        e1 = lambda x: f(x)
        e2 = lambda x: f(x)
        errorsEven.append(max([np.abs(f(x) - poly1(x)) for x in xlin]))
        errorsRunge.append(max([np.abs(f(x) - poly2(x)) for x in xlin]))
        #plt.loglog(xlin, poly1(xlin), label='even' + str(n))
        #plt.loglog(xlin, poly2(xlin), label='runge' + str(n))

    ##plot Graph
    plt.title('Errors: even vs Runge points')
    plt.loglog(numPoints, errorsEven, label='even' + str(n))
    plt.loglog(numPoints, errorsRunge, label='Runge' + str(n))

    #print(max(errorsEven))
    #print(max(errorsRunge))
    plt.legend()
    plt.show()





from numpy.fft import fft
# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    ### get runge points and apply FFT
    y = np.cos((np.pi*np.arange(2*n)) / n)
    samples = f(y)

    coeffs = np.real(fft(samples))[:n+1] / n
    coeffs[0] /=2
    coeffs[-1] /= 2

    return coeffs




def prob6TEEST():
    f = lambda x: -3 + 2 * x ** 2 - x ** 3 + x ** 4
    pcoeffs = [-3, 0, 2, -1, 1]
    ccoeffs1 = np.polynomial.chebyshev.poly2cheb(pcoeffs)
    ccoeffs2 = chebyshev_coeffs(f,4)
    #print(ccoeffs1)
    #print(ccoeffs2)


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    ##load data
    data = np.load('airdata.npy')
    fx = lambda a, b, n: .5 * (a + b + (b - a) * np.cos(np.arange(n + 1) * np.pi / n))
    a, b = 0, 366 - 1 / 24


    ##get points
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)

    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)

    ## create interpolating polynomial
    poly = Barycentric(domain[temp2], data[temp2])

    ###plot graphs
    plt.subplot(1,2,1)
    plt.plot(points, poly.__call__(points))
    plt.title('poly aproximation')

    plt.subplot(1,2,2)
    plt.plot(domain, data)
    plt.title('utah air quality')
    plt.show()

if __name__ == '__main__':
    #prob2()
    #prob3()
    #prob5()
    #teEST4()
    #prob6TEEST()
    #prob7(50)
    pass