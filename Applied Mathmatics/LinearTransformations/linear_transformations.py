# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""
import time
from random import random
import numpy as np
horse = np.load("horse.npy")
from matplotlib import pyplot as plt
# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    T = np.diag([a,b])
    #print(T)
    return np.matmul(T, A)



def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    T = np.mat([[1,a],[b,1]])
    #print(T)
    return np.matmul(T, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    d = 1/(a**2 + b**2)
    T = np.mat([[(a**2 - b**2), (2 * a *b)], [(2 * a * b), (b**2 - a**2)]])*d
    return T@A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    T = ([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos((theta))]])
    return np.matmul(T,A)

# Problem 2
def RotatPE(i, pe):
    return rotate()

def solar_system(T, omega_e, omega_m, x_e, x_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    ##initialize variables
    pe = np.array([[x_e, 0]]).T
    pe0 = np.array([[x_e, 0]]).T
    pePath = pe
    increment = 100
    pm = np.array([[x_m, 0]]).T
    pm0 = np.array([[x_m, 0]]).T
    moonRotate = np.subtract(pm0, pe0)
    pmPath = pm

    for i in range(0,increment):
        #calculate the earths rotation
        pe = rotate(pe, T * omega_e/ increment)
        pePath = np.append(pePath, pe, axis=1)
        #caculate the moones rotation
        moonRotate = rotate(moonRotate, T * omega_m/increment)

        pm = np.add(pe, moonRotate)
        pmPath = np.append(pmPath, pm, axis=1)

    #print(pePath.shape)
    plt.plot(pePath[0], pePath[1], "-b")
    plt.plot(pmPath[0], pmPath[1], "-o")
    plt.show()






def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    MVdata = np.array([[0,0]]).T
    MMdata = np.array([[0,0]]).T

    for i in range(5, 100, 5):
        A = random_matrix(i)
        B = random_matrix(i)
        x = random_vector(i)
        #time first Function
        tstart1 = time.time()
        C = matrix_vector_product(A, x)
        tdif1 = time.time() - tstart1
        MVarray = np.array([[i,tdif1]]).T
        MVdata = np.append(MVdata, MVarray, axis=1)

        #time second function
        tstart2 = time.time()
        D = matrix_matrix_product(A, B)
        tdif2 = time.time() - tstart2
        MMarray = np.array([[i, tdif2]]).T

        MMdata = np.append(MMdata, MMarray, axis=1)
    plt.subplot(121)
    plt.plot(MVdata[0], MVdata[1], "-b")
    plt.title("Matrix-Vector Multiplication")
    plt.xlabel("n")
    plt.ylabel("Seconds")

    plt.subplot(122)
    plt.title("Matrix-Matrix Multiplication")
    plt.plot(MMdata[0], MMdata[1], "-m")
    plt.xlabel("n")
    plt.show()

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    MVdata = np.array([[0,0]]).T
    MMdata = np.array([[0,0]]).T
    MVNPdata = np.array([[0,0]]).T
    MMNPdata = np.array([[0,0]]).T

    for i in range(1, 100, 10):
        A = random_matrix(i)
        B = random_matrix(i)
        x = random_vector(i)

        #Time Matrix_vector function
        tstart1 = time.time()
        C = matrix_vector_product(A, x)
        tdif1 = time.time() - tstart1
        MVarray = np.array([[i,tdif1]]).T
        MVdata = np.append(MVdata, MVarray, axis=1)

        #Time Matrix_matrix function
        tstart2 = time.time()
        D = matrix_matrix_product(A, B)
        tdif2 = time.time() - tstart2
        MMarray = np.array([[i, tdif2]]).T
        MMdata = np.append(MMdata, MMarray, axis=1)

        ## matrix vector multiplication with np.dot
        tstart3 = time.time()
        E = np.dot(A, x)
        tdif3 = time.time() - tstart3
        MVNParray = np.array([[i,tdif3]]).T
        MVNPdata = np.append(MVNPdata, MVNParray, axis=1)

        ## matrix mutiplication with np.dot
        tstart4 = time.time()
        F = np.dot(A, B)
        tdif4 = time.time() - tstart4
        MMNParray = np.array([[i, tdif4]]).T
        MMNPdata = np.append(MMNPdata, MMNParray, axis=1)

    ###plot the linear scale plot
    plt.subplot(121)

    plt.plot(MVdata[0], MVdata[1], "-b")
    plt.plot(MMdata[0], MMdata[1], "-m")
    plt.plot(MVNPdata[0], MVNPdata[1], "-r")
    plt.plot(MMNPdata[0], MMNPdata[1], "-g")

    plt.title("Matrix-Matrix Multiplication")
    plt.xlabel("n")
    plt.ylabel("Seconds")



## plot the log scale plot
    plt.subplot(122)
    plt.plot(MVdata[0], MVdata[1], "-b")
    plt.plot(MMdata[0], MMdata[1], "-m")
    plt.plot(MVNPdata[0], MVNPdata[1], "-r")
    plt.plot(MMNPdata[0], MMNPdata[1], "-g")
    plt.title("Log Scale")
    plt.xlabel("n")
    plt.xscale("log")
    plt.yscale("log")


    plt.show()

def createHorses():
    #draw the regular horse
    plt.subplot(231)
    plt.plot(horse[0], horse[1],"k.")
    plt.axis([-1, 1, -1, 1])

    #draw the steched horse
    strechHorse = stretch(horse, .5, 1.2)
    plt.subplot(232)
    plt.plot(strechHorse[0], strechHorse[1],"k.")
    plt.axis([-1, 1, -1, 1])

    ##draw the shear horse
    shearHorse = shear(horse, .5, 0)
    plt.subplot(233)
    plt.plot(shearHorse[0], shearHorse[1],"k.")
    plt.axis([-1, 1, -1, 1])


    ##draw the reflect horse
    reflectHorse = reflect(horse, 0, 1)
    plt.subplot(234)
    plt.plot(reflectHorse[0], reflectHorse[1],"k.")
    plt.axis([-1, 1, -1, 1])


    ##draw the reflect horse
    rotateHorse = rotate(horse, np.pi/2)
    plt.subplot(235)
    plt.plot(rotateHorse[0], rotateHorse[1],"k.")
    plt.axis([-1, 1, -1, 1])


    ##draw the reflect horse
    compHorse = stretch(shear(reflect(rotate(horse, np.pi/2),0,1),.5,0), .5, 1.2)
    plt.subplot(236)
    plt.plot(compHorse[0], compHorse[1],"k.")
    plt.axis([-1, 1, -1, 1])
    plt.show()


if __name__ == '__main__':
    #createHorses()
    #solar_system(np.pi * 1.5, 1, 13, 10, 11)
    #prob4()
    A = np.mat([[1,2,3], [3,5,8]])
    b = 10
    a = 3
    #print(reflect(A, a, b ))

    #print(reflect(np.array([[2, 3], [5, 6]]), 10, 3))

   # prob3()
    pass

