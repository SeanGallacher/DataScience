# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse
# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

##helper functions
def runIterMethod(A, b, tol=1e-8, maxiter=100, plot=False, method='Jacobi', omega=1, rCon=False,rIts=False):
    m, n = A.shape
    if not sparse.issparse(A):
        D = np.diag(A)
    else:
        D = A.diagonal()
    its = 0
    x, xp = np.zeros(n), np.ones(n)
    if method == 'Jacobi':
        get_nextx = lambda x: x + (b - A @ x) / D
    if method == 'Gauss':
        def get_nextx(x):
            for i in range(len(x)):
                x[i] = x[i] + omega*(b[i] - A[i].T @ x) / D[i]
            return x
    if method == 'Sparse':
        Arows = []
        for i in range(A.shape[0]):
            rowstart = A.indptr[i]
            rowend = A.indptr[i + 1]
            Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]
            Arows.append(Aix)
        def get_nextx(x):
            for i in range(len(x)):
                x[i] = x[i] + (b[i] - Arows[i]) / D[i]
            return x
    aproxs = []
    while its < maxiter and la.norm(xp - x, ord=np.inf) > tol:
        xp = x

        if method!='Sparse':
            aproxs.append(la.norm(A @ x - b, ord=np.inf))
        x = get_nextx(x)
        its += 1
    plt.semilogy([i for i in range(len(aproxs))], aproxs, label='aproximations')
    plt.title('Conergence of Jacobian')
    plt.show()
    if omega !=1:
        return x, its
    return x
# Problems 1 and 2


def jacobi(A, b, tol=1e-8, maxiter=100, plot=False, method='Jacobi'):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    return runIterMethod(A,b)




# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    return runIterMethod(A, b, tol=tol, maxiter=maxiter, plot=plot, method='Guass')



import time
# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    return runIterMethod(A, b, tol=tol, maxiter=maxiter, method='Sparse', ts=time.time())

# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    return runIterMethod(A, b, tol=tol, maxiter=maxiter, method='Guass', omega=omega)

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """

    diag = []
    n4 = []
    n1 = []
    ##creating the Diagnols for B
    for i in range (0,n):
        n4.append(-4)
        if i !=0:
            n1.append(1)
    offsets = [-1,0,1]
    ##creating the diagnol list forB
    diag.append(n1)
    diag.append(n4)
    diag.append(n1)

    #creating B
    B = sparse.diags(diag, offsets, shape=(n,n))
    I = np.eye(n)

    ##Creating A
    A = sparse.block_diag([B] * n)
    A.setdiag(1, n)
    A.setdiag(1, -n)

    return A
# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=1000, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    A = prob5(n)
    t = np.zeros(n)
    t[0],t[-1] = -100, -100
    b = np.tile(t, n)
    u = sor(A,b,omega,tol=tol, maxiter=maxiter)
    U = np.reshape(u, (n,n))
    if plot:
        plt.pcolormesh(U, colormap='coolwarm')
        plt.title('HEAT TRANSFSER')
        plt.show()
    return u, True, 19

if __name__ == '__main__':
    b = np.random.random(10)
    A = diag_dom(10)
    A = sparse.csr_matrix(diag_dom(5000))
    b = np.random.random(5000)
    x = gauss_seidel_sparse(A,b)
    print(x)
    print(A.dot(x))
    print(b)
    print(np.allclose(A@x,b))
    hot_plate(10, 1)


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    its = []
    ws = [1+ w/20 for w in range(0,20)]
    for w in ws:
        u,c,it = hot_plate(20,w )
        its.append(it)
    plt.plt(ws, its)
    plt.show()
