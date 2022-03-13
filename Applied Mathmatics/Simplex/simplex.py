"""Volume 2: Simplex

<Name>
<Date>
<Class>
"""

import numpy as np
from numpy import linalg as la

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        ##load in the data
        self.c = c
        self.A = A
        self.b = b
        self.n = A.shape[1]
        if np.any(b < 0):
            raise ValueError('Negative b value')

    # Problem 2

    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        m, n = A.shape
        Ahat = np.column_stack([A,np.eye(m)])
        chat = np.append(c,np.zeros(m))

        ###create the dictionary
        Dright = np.row_stack([chat,-Ahat])
        Dleft = np.hstack([0,b])
        D = np.column_stack([Dleft,Dright])
        self.D = D
        self.rowsDone = []


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        #print(self.rowsDone)
        m, n = self.D.shape
        ###find the piviot column
        for c in range(1,n):
            if c in self.rowsDone:
                continue
            col = self.D[:,c]
            #print("c " + str(c))
            if col[0] < 0:
                self.col = c
                self.rowsDone.append(c)
                return c
        self.col = 0
        return 0


    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        self.b = self.D[1:,0]
        if index == 0: return 0
        ratios = np.divide(-self.b, self.D[1:, index])
        ### find the piviot row
        for i, r in enumerate(ratios):
            if self.D[i+1, index] > 0:
                ratios[i] = np.inf
            if r < 0:
                ratios[i] = np.inf
        self.rp = np.argmin(ratios) + 1
        return self.rp


    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        ### turn row into 1
        m, n = self.D.shape
        ##check for unbounded
        if self.col == 0 or np.all(self.D[:,self.col] > 0):
            raise ValueError('unbounded')
        ##row mutiplication
        self.D[self.rp,:] = -self.D[self.rp,:]/self.D[self.rp,self.col]
        for r in range(m):
            if r == self.rp:
                continue

            k = self.D[r,self.col]
            self.D[r,:] += k*self.D[self.rp,:]

        return self.D


    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        m, n = self.D.shape
        indIn = []
        depIn = []
        ### go through pivots and make final dict
        for i in range(1,n):
            if np.all(self.D[0,1:] >= 0):
                break
            self._pivot_row(self._pivot_col())
            self.pivot()

        #### return indecies with right comuns
        A = []
        b = self.D[1:,0]
        for c in range(1,n):
            if self.D[0,c] ==0:
                A.append(self.D[1:,c])
                depIn.append(c-1)
            else:
                indIn.append(c-1)



        A = np.column_stack(A)

        values = la.solve(-A,b)
        vlen = len(values)
        dep = dict(zip(depIn, values))
        ind = dict(zip(indIn,[0 for i in range(len(indIn))]))
        return (self.D[0,0], dep,ind)



# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    ##load files
    data = np.load(filename)
    A = np.row_stack([data['A'],np.eye(len(data['d']))])
    c = -data['p']
    m = np.hstack([data['m'],data['d']])

    ###solve using simplex
    SS = SimplexSolver(c,A,m)
    SS._generatedictionary(c,A,m)
    sol = SS.solve()
    newDict = {**sol[1], **sol[2]}
    values = np.array([newDict[i] for i in range(len(c))])
    #values = list(sol[1].values()) + list(sol[2].values())
    return values


def hwTestFunctions(c, A, b):
    SS = SimplexSolver(c, A, b)
    SS._generatedictionary(c, A, b)
    print(SS.solve())
    print(SS.D)

if __name__ == '__main__':
    A = np.array([[1,-1],[3,1],[4,3]])
    b = np.array([2,5,7])
    c = np.array([-3,-2])
    SS = SimplexSolver(c,A,b)
    #SS._generatedictionary(c,A,b)
    #print(SS.D)
    #print(SS._pivot_col())
    #print(SS._pivot_row(SS._pivot_col()))
    #print(SS.pivot())
    #print(SS.solve())
    #print(prob6())

    A = np.array([[2,-3],[1,-6],[1,1]])
    b = np.array([4,1,6])
    c = np.array([-5,4])

    c = np.array([-4, -3])
    A = np.array([[15, 10], [2, 2], [0, 1]])
    b = np.array([1800, 300, 200])

    c = np.array([-10, 57,9,24])
    A = np.array([[.5,-1.5,-.5,1 ], [.5,-5.5,-2.5,9], [1, 0,0,0]])
    b = np.array([1j, .00001, 1])

    c = np.array([-3, -1])
    A = np.array([[1,3 ], [2,3],[1,-1]])
    b = np.array([15,18,4])


    A = np.array([[1, 3], [2, 3], [1, -1]])
    b = np.array([15, 18, 4])
    c = np.array([-3, -1])


    c = np.array([-4, -6])
    A = np.array([[-1, 1], [1, 1], [2, 5]])
    b = np.array([11, 27, 90])

    c = np.array([-2, 1])
    A = np.array([[1,-2], [1, 1], [1, -1]])
    b = np.array([1,3,1])

    c = np.array([3., 2.])
    b = np.array([2., 5., 7.])

    A = np.array([[1., -1.],
                  [3., 1.],
                  [4., 3.]])

    hwTestFunctions(c, A, b)



