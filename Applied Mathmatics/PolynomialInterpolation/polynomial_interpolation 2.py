# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt
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

    print(L_num)
    L = L_num.T/L_demon

    p = np.sum(yint*L, axis =1)
    return p


def prob2():
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
        self.X = xint
        self.Y = yint
        self.n = len(xint)

        self.Wjs = np.array([1/np.prod([xint[j] - xint[k] for k in np.delete(np.array(range(self.n)), j)]) for j in range(self.n)])

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """

        w = lambda x: np.product([x-xj for xj in self.X])

        p = lambda x: w(x)* np.sum([self.Wjs[j]*self.Y[j] / (x-self.X[j]) for j in range(self.n)])

        return np.array([p(x) for x in points])
    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        self.Y.append(yint)
        w = lambda x: np.product([x - xj for xj in self.X])
        for i in range(len(xint)):
            self.X.append(w(xint[i]))





def prob3():
    x = np.linspace(-1,1, 11)
    x2 = np.linspace(-1, 1, 100)
    f = lambda x: 1/(1+(25*x**2))
    baram = Barycentric(x,f(x))
    p = baram.__call__(x2)

    plt.plot(x2, p)

    plt.plot(x2,f(x2))
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
    f = lambda x: 1 / (1 + (25 * x ** 2))
    errorsEven = []
    errorsRunge = []
    a = -1
    b = 1
    r = lambda x: .5 *(a+b + (b-a)*np.cos(x*np.pi/n))

    numPoints = [2**a for a in range(2,9)]

    xlin = np.linspace(-1,1,400)

    for n in numPoints:

        ptsEven = np.linspace(-1,1, n)
        ptsRunge = np.array([r(x) for x in ptsEven])

        ##figure out why poly is not working
        poly1 = BarycentricInterpolator(ptsEven)
        poly2 = BarycentricInterpolator(ptsRunge)


        e1 = lambda x: f(x)
        e2 = lambda x: f(x)
        errorsEven.append(max([np.abs(e1(x)) for x in xlin]))
        errorsRunge.append(max([np.abs(e2(x)) for x in xlin]))

    plt.loglog(numPoints, errorsRunge, label='runge')
    plt.loglog(numPoints, errorsEven, label='even')
    plt.legend()
    plt.show()





import scipy.fftpack as fft
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
    xRunge = [x*np.pi/n for x in range(n +1)]
    ak = 2*fft.fft(xRunge).real*(1/(n))
    ak[0] /=2
    ak[-1] /=2
    if False:
        T = [lambda x: 1, lambda x: x]
        for i in range(1, n):
            f1 = T[i]
            f2 = T[i-1]
            T.append(lambda x: 2 * x * f1(x) - f2(x))

        p = lambda x: np.sum(ak[j] * T[j](x) for j in range(n+1))

    return ak





def prob6TEEST():
    f = lambda x: -3 + 2 * x ** 2 - x ** 3 + x ** 4
    pcoeffs = [-3, 0, 2, -1, 1]
    ccoeffs1 = np.polynomial.chebyshev.poly2cheb(pcoeffs)
    ccoeffs2 = chebyshev_coeffs(f,5)
    print(ccoeffs1)
    print(ccoeffs2)


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load('airdata.npy')
    fx = lambda a, b, n: .5 * (a + b + (b - a) * np.cos(np.arange(n + 1) * np.pi / n))
    a, b = 0, 366 - 1 / 24

    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)

    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)

    poly = Barycentric(domain[temp2], data[temp2])

    plt.subplot(1,2,1)
    plt.plot(points, poly.__call__(points))
    plt.title('poly aproximation')

    plt.subplot(1,2,2)
    plt.plot(domain, data)
    plt.title('utah air quality')
    plt.show()

if __name__ == '__main__':
    #prob2()
    prob3()
    #prob5()
    #prob6TEEST()
    #prob7(50)