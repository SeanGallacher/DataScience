# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name>
<Class>
<Date>
"""
import cvxpy as cp
import numpy as np


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3, nonneg=True)
    c = np.array([2,1,3])
    objective = cp.Minimize(c.T@x)
    G = np.array([[1,2,0],[0,1,-4],[-2,-10,-3]])
    b = np.array([3,1,-12])
    constraints = [G@x <=b]
    problem  = cp.Problem(objective, constraints)
    sol = problem.solve()
    return x.value, sol






# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    m,n = A.shape
    x = cp.Variable(n, nonneg=True)
    constraints = [A@x == b]
    objective = cp.Minimize(cp.norm(x, 1))
    problem = cp.Problem(objective, constraints)
    sol = problem.solve()
    return x.value, sol




# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    p = cp.Variable(6, nonneg=True)
    G = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])
    gb = np.array([7,2,4])
    A = np.array([[1,0,1,0,1,0],[0,1,0,1,0,1]])
    ab = np.array([5,8])

    c = np.array([4,7,6,8,8,9])
    objective = cp.Minimize(c.T@p)
    constraints = [G@p <=gb, A@p ==ab]
    problem = cp.Problem(objective, constraints)
    sol = problem.solve()
    return p.value, sol


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    g = lambda x,y,z: (3 / 2) *x ** 2 + 2*x*y + x*z + 2*y ** 2 + 2*y*z + (3 / 2)*z ** 2 + 3*x + z
    x = cp.Variable(3)
    a,b,c,d,e,f = 3/2, 1, 1/2, 2, 1, 3/2
    Q = 2* np.array([[a,b,c],[b,d,e],[c,e,f]])
    r = np.array([3,0,1])
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    sol = problem.solve()
    return x.value, sol


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    m, n = A.shape
    x = cp.Variable(n, nonneg=True)
    ones = np.ones(n)
    constraints = [ones.T @x == 1 ]
    objective = cp.Minimize(cp.norm(A@x-b,2))
    problem = cp.Problem(objective, constraints)
    sol = problem.solve()
    return x.value, sol


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    FD = np.load('food.npy', allow_pickle=True).T
    x = cp.Variable(18,nonneg=True)
    objective = cp.Minimize(FD[0]@x)
    b = np.array([2000,65,50,-1000,-25,-46])
    M = FD[1]*FD[2:]
    M[-3:] *=-1
    constraints = [M@x <= b]
    problem = cp.Problem(objective, constraints)
    sol = problem.solve()
    print('EAT PATATOS, MILK, and a little cheese')
    return x.value, sol

if __name__ == '__main__':
    pass
