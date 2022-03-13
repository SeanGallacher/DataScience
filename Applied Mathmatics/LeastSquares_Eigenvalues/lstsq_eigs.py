# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name
<Class
<Date
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg
import cmath

import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from time import time
import time
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode='economic')
    x = la.solve_triangular(R, Q.T@b)
    if not np.array_equal(A@x, b):
        #print(A@x)
        #print(b)
        pass
    return x


# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    housing = np.load('housing.npy')
    ##prep the data
    A = np.array([housing[:,0]]).T
    ones = np.ones((1, len(housing[:,0].T))).T
    A = np.hstack((A,ones))
    b = np.array([housing[:,1]]).T
    x = least_squares(A,b)

    housing = housing.T
    housing = housing.astype('float')

    plt.scatter(housing[0],housing[1], c="green")
    X = np.zeros_like(housing)
    m, n = housing.shape

    ##plot the data
    for i in range(n):
        ## x cordinate
        X[0][i] = housing[0][i]
        ## y cordinate
        X[1][i] = housing[0][i] * x[0] + x[1]

    plt.plot(X[0], X[1], "-r")
    plt.title("Matrix Operations")
    plt.xlabel("year")
    plt.ylabel("husing price")

    plt.show()




# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    subplotList= [221,222,223,224]
    subplotIndex = -1
    for i in range(3,13,3):
        subplotIndex+=1
        ##get least Squares
        housing = np.load('housing.npy')
        A = np.vander(housing[:, 0],i+1)
        b = np.array(housing[:, 1]).T
        x = least_squares(A, b)
        ###get Equation
        Xeq = np.poly1d(x)
        x_vals = np.linspace(0,16,1000)
        y_vals = Xeq(x_vals)

        housing = housing.T

        ## plot the data
        plt.subplot(subplotList[subplotIndex])
        plt.scatter(housing[0],housing[1], c="blue")
        plt.plot(x_vals,y_vals, '-g')
        plt.title("Poly fit degree: " + str(i))
        plt.xlabel("year")
        plt.ylabel("housing price")

    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)
    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")
    plt.show()



# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    xk, yk = np.load("ellipse.npy").T

    ### Calculate least Sqaures for an Ellipse
    A = np.column_stack(( xk ** 2,  xk,  xk * yk,  yk, yk ** 2))
    b = np.ones_like(xk)
    a,b,c,d,e = la.lstsq(A, b)[0]
    plt.plot(xk, yk, 'k*')
    plt.title("Ellipse")
    plot_ellipse(a,b,c,d,e)


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    ### Use the power method to find an eigenvalue
    def procedure(A):
        m,n = A.shape
        x = [None] * (N + 1)
        x0 = np.random.random((n,1))
        x0 /= la.norm(x0)
        x[0] = (x0)
        for k in range(0,N):
            #print(A.shape)
           # print(x[k].shape)
            x[k +1 ] = A@x[k]
            x[k + 1]/= la.norm(x[k+1])
            if la.norm(x[k+1] - x[k]) < tol:
                k = N
        return x[N].T@A@x[N], x[N]

    ###Make sure the eighenValues are Correct
    eigs, vecs = la.eig(A)
    loc = np.argmax(eigs)
    lamb, x = eigs[loc], vecs[:, loc]
    #assert(np.allclose(A @ x, lamb * x))
    return procedure(A)

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = A.shape
    ### Create S
    S = la.hessenberg(A)
    for k in range(N):
        Q, R = la.qr(S)
        S = R@Q
    eigs = np.zeros(n)

    i = 0

    while i < n:
        ### if Si is one by one
        if i == n-1:
            eigs[i] = (S[i][i])
        elif np.abs(S[i+1][i]) < tol:
            eigs[i] = (S[i][i])
        ### if Si is two by two
        else:
            a, b, c, d = S[i:i+2,i:i+2].ravel()
            A = 1
            B = -a-d
            C = a*d - b*c
            val1 = (-B + cmath.sqrt(B**2 - 4 * A * C))/(2 * A)
            val2 = (-B - cmath.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            #print(val1)
            eigs[i] = val1
            eigs[i+1] = val2
            i = i +1
        i = i + 1

    return eigs

def tesstProb5():
    A = np.array([[0.5364795, 0.45771575, 0.86882758, 0.28528139, 0.01027844],
                  [0.74756167, 0.67846898, 0.27875227, 0.0287938, 0.113242],
                  [0.79402049, 0.90306769, 0.12510102, 0.44017928, 0.44274237],
                  [0.3901788, 0.02779846, 0.200347, 0.91972721, 0.57303935],
                  [0.02678807, 0.39017548, 0.60900494, 0.51795614, 0.4712653]])
    print(power_method(A))
    print( (2.1650302166455964,
    np.array([0.46678674, 0.37440866, 0.5310185 , 0.43174126, 0.41658989])))

def other():
    A6 = np.random.random((6,6))
    AS6 = A6 + A6.T
    A5 = np.random.random((5,5))
    AS5 = A5 + A5.T

    print(la.eig((AS6)))
    print(qr_algorithm(AS6))

    print(la.eig((AS5)))
    print(qr_algorithm(AS5))

def TryProb6():
    A = np.random.random((7, 7))
    A_sym = A + A.T
    print(A_sym, '\n\n')
    print("Mine: ", qr_algorithm(A_sym))
    print("Scipy: ", la.eig(A_sym)[0])
    print(np.allclose(np.sort(qr_algorithm(A_sym)), np.sort(la.eig(A_sym)[0])))
if __name__ == '__main__':
    A = np.array([[2, 7],
                  [3, -2]], dtype=np.float)
    b = np.array([[1,1]])
    #least_squares(A,b)
    #ellipse_fit()
    #power_method(A,20)
    #tesstProb5()
    #TryProb6()
    A = np.array([[3]])
    #print(qr_algorithm(A))
    A2 = np.array([[1, 0], [0, 2]])
    #print(qr_algorithm(A2))
    A3 = np.array([[2, 3], [4, 5]])
    #print(qr_algorithm(A3))
    A = np.array([[0.92290788, 0.05667364, 0.34250155, 0.83995084, 0.77660243,
                   0.08021165],
                  [0.90673232, 0.86903985, 0.48835738, 0.29807951, 0.36728684,
                   0.65988196],
                  [0.6024126, 0.15685944, 0.52362323, 0.62144535, 0.63714732,
                   0.33921522],
                  [0.9098743, 0.65716983, 0.21670957, 0.36880413, 0.83881214,
                   0.63232412],
                  [0.70780166, 0.67741961, 0.39962132, 0.66939116, 0.32739838,
                   0.27707362],
                  [0.32694547, 0.83418833, 0.10122551, 0.91047088, 0.44474892,
                   0.08358557]])
    #print(np.sort(qr_algorithm(A)))
    #print(np.sort(np.array([-0.73488734+0.j       , -0.0404974 +0.j       ,
    #    0.11231013-0.2084547j,  0.11231013+0.2084547j,
    #    0.4607786 +0.j       ,  3.18534492+0.j       ])))

    Astoc = np.array([[.7,.6],[.3,.4]])
    print(np.allclose(A.sum(axis=0), np.ones(A.shape[1])))
    print(Astoc.sum(axis=0))
    print(np.ones(Astoc.shape[1]))


