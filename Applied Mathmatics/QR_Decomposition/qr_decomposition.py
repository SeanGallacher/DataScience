# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name>
<Class>
<Date>
"""
import math

import numpy as np
from time import time
import time
from matplotlib import pyplot as plt
from scipy import linalg as la


# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    ##set up variables
    Anorm = la.norm(A)
    m, n = A.shape
    Q = A.copy()
    R = np.zeros((n, n))

    ###Do the Gram smit QR decomposition
    for i in range(0, n):
        R[i, i] = la.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, j].T, Q[:, i])
            Q[:, j] = Q[:, j] - np.dot(R[i, j], Q[:, i])

    ### check to make the QR decom worked
    if not np.array_equal(np.matmul(Q, R), A):
        # print("INCORRECT MATRIX")
        # print(np.matmul(Q, R))
        pass
    return Q, R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    ###Calculate the determinant
    Q, R = qr_gram_schmidt(A)
    return math.prod(np.diag(R))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    ##Set up variables
    m, n = A.shape
    Q, R = qr_gram_schmidt(A)
    y = np.dot(Q.T, b)
    x = [None] * n

    ### Back substitue
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(n - 1, i, -1):
            x[i] -= x[j] * R[i, j]
        x[i] /= R[i, i]
    #print(np.dot(A, x))
    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    ##Set up variables
    sign = lambda x: 1 if x >= 0 else -1
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)

    ### perform the qr householder decomposition
    for k in range(n):
        u = R[k:, k].copy()
        u[0] = u[0] + sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        R[k:, k:] = R[k:, k:] - np.outer(2 * u, np.dot(u.T, R[k:, k:]))
        Q[k:, :] = Q[k:, :] - np.outer(2 * u, np.dot(u.T, Q[k:, :]))

    #print(Q.T@R)
    return Q.T, R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    ## set up variables
    sign = lambda x: 1 if x >= 0 else -1
    m,n = A.shape
    H = A.copy()
    Q = np.eye(m)

    #Perform the hessenberg decomposition
    for k in range(0,n-2):
        u = H[k+1:,k].copy()
        u[0] = u[0] + sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        H[k+1:,k:] -= np.outer(2 * u, np.dot(u.T, H[k+1:, k:]))
        H[:,k+1:] -= 2*np.outer((np.dot(H[:,k+1:],u)),u.T)
        Q[k+1:,:] -= np.outer(2*u,np.dot(u.T,Q[k+1:,:]))

    #print(Q @ H @ Q.T)
    return(H,Q.T)

if __name__ == '__main__':
    A = np.array([[2, 7, 1],
                  [3, -2, 9],
                  [1, 5, 3]], dtype=np.float)
    b = np.array([1, 2, 3])
    #qr_gram_schmidt(A)
    #print(qr_gram_schmidt(A)[0])
    #print(qr_gram_schmidt(A)[1])
    # print(la.det(A))
    #print(abs_det(A))
    #print(solve(A, b))

    #qr_householder(A)
    #hessenberg(A)
