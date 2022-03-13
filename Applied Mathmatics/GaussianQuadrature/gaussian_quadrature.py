# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
<Name>
<Class>
<Date>
"""
from scipy.integrate import nquad
import numpy as np
from scipy.integrate import quad
from scipy import linalg as la
from scipy.stats import norm
from matplotlib import pyplot as plt
class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    L = "legendre"
    C = "chebyshev"
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        ## check the input the legendary user
        if polytype != 'legendre' and polytype != 'chebyshev':
            raise ValueError("Incorrect polytype")
        ##intialize based on polytype
        if polytype == 'legendre':
            self.w = lambda x: 1
            self.C = False
        else:
            self.w = lambda x: 1/np.sqrt(1-x**2)
            self.C = True
        self.wr = lambda x: 1/self.w(x)

        self.xis, self.wis = self.points_weights(n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        ### Create the jacobean matrix
        if self.C:
            Bk = [1/4 for k in range(n+1)]
            Bk[1] = 1/2
            u = np.pi
        else:
            Bk = [(k**2)/(4*k**2-1) for k in range(n+1)]
            u = 2

        D1 = np.diag(np.sqrt(Bk[1:-1]), k=-1)
        D2 = np.diag(np.sqrt(Bk[1:-1]), k=1)
        J = D1+D2
        #print(J)

        eigs, vectors = la.eig(J)
        self.n = n



        w = np.array([u * val**2 for val in vectors[0]]).real


        return eigs.real, w
    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        ### look at the comment above
        g = lambda x: f(x)*self.wr(x)
        gxis = g(self.xis) #[g(xi) for xi in self.xis]
        return self.wis @ gxis  #np.real(np.sum(gxis[i] * self.wis[i] for i in range(len(self.xis))))
    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        ### return intregral
        h = lambda x: f((b-a)*x/2 + (a+b)/2)
        return ((b-a)/2)*self.basic(h)
    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        ##calculate 2 d interal
        n = self.n
        h = lambda x: f( (b1-a1)*x[0]/2 + (a1+b1)/2, (b2-a2)*x[1]/2 + (a2+b2)/2 )
        g = lambda x: h((x[0], x[1]))*self.wr(x[0])*self.wr(x[1])

        wwg = [[self.wis[i]*self.wis[j]*g((float(self.xis[i]),float(self.xis[j])) ) for i in range(n)] for j in range(n)]
        return (b1-a1)*(b2-a2)*np.sum(wwg)/4

# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    ##initlize functions
    f =lambda x: (1 / np.sqrt(2*np.pi))*np.exp( ((-x ** 2) / 2))

    exact = norm.cdf(2) - norm.cdf(-3)
    errorsLeg = []
    errorsChev = []
    ns = [n for n in range(5,55,5)]

    ### calculate the erros
    for n in ns:
        GQ = GaussianQuadrature(n)
        errorsLeg.append(np.abs(GQ.integrate(f,-3.,2.) - exact))
        GQC = GaussianQuadrature(n, 'chebyshev')
        errorsChev.append(np.abs(GQC.integrate(f, -3., 2.) - exact))


    errorQuad = quad(f,-3,2)- exact

    ###Plot the functions
    plt.semilogy([5, 50], [errorQuad, errorQuad], label='exact', marker='.')
    plt.semilogy(ns, errorsLeg, label='legengre', marker='.')
    plt.semilogy(ns, errorsChev, label='chebyshev', marker='.')
    plt.xlabel('n points')
    plt.ylabel('error')
    plt.title('Discovery of Guassian Quadature Method Errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    GQ = GaussianQuadrature(5)
    # Integrate f(x) = 1 / sqrt(1 - x**2) from -1 to 1.
    f = lambda x: 1 / np.sqrt(1 - x ** 2)
    print(quad(f, -1, 1)[0])
    print(GQ.basic(f))
    print(GQ.wis)
    prob5()


    # Integrate f(x,y) = sin(x) + cos(y) over [-10,10] in x and [-1,1] in y.
    f = lambda x, y: np.sin(x) + np.cos(y)
    print(nquad(f, [[-10, 10], [-1, 1]])[0])

    GQ = GaussianQuadrature(13, 'chebyshev')
    print(GQ.integrate2d(f,-10,10,-1,1))


    f = lambda x: x**2
    GQ = GaussianQuadrature(30)

    print(GQ.basic(f))
    print(GQ.integrate(f,1,4))

