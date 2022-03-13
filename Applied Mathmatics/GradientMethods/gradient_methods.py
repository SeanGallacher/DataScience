# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name>
<Class>
<Date>
"""


# Problem 1
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from numpy import linalg as la
from autograd import grad
from autograd import grad as anp

def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    it = 1
    change = 1.

    while it < maxiter and change > tol:
        ###CALULATE NEW XK
        yk = Df(x0)
        a = lambda alp: f(x0 - alp * yk.T)
        alpha = opt.minimize_scalar(a).x
        xk = x0 - alpha * yk
        ##UPDATE VALUES
        change = np.linalg.norm(yk,ord=np.inf)  # np.max([xk[i]-x0[i] for i in range(len(xk))])
        x0 = xk
        it += 1

    return x0, change < tol, it


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    ##INITIALIZE VARS
    rk = Q@x0 - b
    dk = -rk

    k = 0
    n = b.shape[0]
    xk = x0

    while la.norm(rk) >= tol and k < n:
        ak = (rk.T @ rk)/(dk.T @ Q @ dk)
        ##CALULATE NEW XK
        xkn = xk + ak*dk
        rkn = rk + ak* Q @ dk
        Bk = (rkn.T@rkn)/(rk.T@rk)
        dkn = -rkn + Bk*dk
        ##UPDATE VALUES
        k +=1
        dk = dkn
        xk = xkn
        rk = rkn
    return xkn, la.norm(rkn) < tol, k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    ##INITALIZE VARIABLES
    xk = x0
    rk = -df(x0).T
    dk = rk

    g = lambda alp: f(xk + alp * dk)
    ak = opt.minimize_scalar(g).x

    xk = x0+ak*dk
    k=1

    while la.norm(rk) > tol and k < maxiter:
        ###CALCUATE NEW XK
        rkn = -df(xk).T
        Bk = (rkn.T@rkn)/(rk.T@rk)
        dkn = rkn + Bk * dk
        g = lambda alp: f(xk + alp * dkn)
        ak = opt.minimize_scalar(g).x
        print(ak,Bk,dkn, rkn)
        xk = xk + ak*dkn
        ##UPDATE VALUES
        rk = rkn
        dk = dkn
        k += 1

    return xk, la.norm(rkn) < tol, k


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    ###LOAD DATA
    data = np.loadtxt(filename)
    b = data[:,0].copy()
    data[:,0] = np.ones_like(data[:,0])
    ### FIND THE MINIMIZER
    A = data
    Q = A.T@A
    b = A.T@b
    x = conjugate_gradient(Q,b, x0)[0]
    return x


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        ###FIND THE OPTIMAL B
        print(x,y,guess)
        f = lambda B: np.sum(np.array([np.log(1 + np.exp(-(B[0]+B[1]*x[i]))) + (1-y[i])*(B[0] + B[1]*x[i]) for i in range(len(x))]))

        self.B = opt.minimize(f, guess).x


    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        ##PREDICT USING LR
        s = lambda x: 1/(1 + np.exp(-(self.B[0] + self.B[1]*x)))
        return s(x)

# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    ###LOAD DATA
    data = np.load('challenger.npy')
    temp = data[:,0]
    oDamage = data[:,1]

    LG = LogisticRegression1D()
    LG.fit(temp,oDamage, guess)

    xlin = np.linspace(30,100,200)
    ###PREDICT EXPLOSIONS
    curve = LG.predict(xlin)
    plt.plot(xlin,curve, label='prediction curve')

    plt.scatter(temp,oDamage,label='historical values')
    ###PLOT THE GRAPH
    plt.scatter(31,LG.predict(31),label ='P damage at launch',color='red')
    plt.legend()
    plt.title('predicting spaceship explosions')
    plt.show()

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
if __name__ == '__main__':
    f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
    df = lambda x: np.array([4*x[0]**3 , 4*x[1]**3 , 4*x[2]**3])
    #print(steepest_descent(f, df, np.array([5, 5,1]), maxiter=10000))
    #print(steepest_descent(opt.rosen,opt.rosen_der,np.array([5,5]), maxiter=10000))

    #opt.fmin_cg(opt.rosen, np.array([-2, 2]), fprime=opt.rosen_der)
    #print(nonlinear_conjugate_gradient(opt.rosen, opt.rosen_der,np.array([-2, 2])))
    ###test prob 2
    A = np.random.random((4,4))
    #while not is_pos_def(A):
    #    A = np.random.random((4,4))
    b = np.random.random(4)
    if False:
        print()
        print('PROB 2')
        print(conjugate_gradient(A.T@A,b,np.array([1,2,3,4])))
        print(la.solve(A.T@A,b))


    #Q = np.array([[2, 0], [0, 4]])
    #b = np.array([1, 8])
    #x = np.array([1, 1])
    #print(conjugate_gradient(Q, b, x))

    #print(prob4())
    #prob6()
