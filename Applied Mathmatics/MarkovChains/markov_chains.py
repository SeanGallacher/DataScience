# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
import numpy as np

class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        ###Set up the Markov Martix and dictionary
        m,self.n = A.shape
        if not np.allclose(np.sum(A,axis=0), np.ones(self.n)):
            raise ValueError("Not column Stochastic")
        self.labelDict = {}
        self.A = A
        if states is None:
            states = [i for i in range(self.n)]
        for i in range(len(states)):
            self.labelDict[states[i]] = i

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        ###Create a pair key and val list and draw from a multinomial distrution to get the next state
        key_list = list(self.labelDict.keys())
        val_list = list(self.labelDict.values())
        return key_list[val_list.index(np.argmax(np.random.multinomial(1, self.A[:,self.labelDict[state]])))]

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        ### call Trasition N - 1 times and return the states
        stateList = [start]
        for i in range(N-1):
            start = self.transition(start)
            stateList.append(start)
        return stateList

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        ### run through the path until the end state is found
        stateList = []
        while True:
            stateList.append(start)
            start = self.transition(start)
            if start == stop:
                stateList.append(start)
                return stateList


    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        ###Initialize a random state
        x0 = np.random.random((self.n,))
        xkb = x0/np.sum(np.abs(x0))

        ###calculate xk in steady state
        for i in range(maxiter):
            xk = self.A@xkb
            if np.sum(np.abs(xk-xkb)) < tol:
                return xk
            xkb = xk
        raise ValueError("too many iterations")



class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        ###read in information
        f = open(filename)
        words = set()
        for line in f.readlines():
            # reading each word
            for word in line.split():
                # adding word to set
                words.add(word)
        words = list(words)
        words.insert(0, '$tart')
        words.append('$top')
        n = len(words)
        T = np.zeros((n,n))

        f.close()
        f = open(filename)
        ###Populate the transition matrix
        for line in f.readlines():
            ###create sentence list
            sentence = line.split()
            sentence.insert(0, '$tart')
            sentence.append('$top')

            for i in range(0, len(sentence)-1):
                ###grab the index for X and Y
                wordIndexX = words.index(sentence[i])
                wordIndexY = words.index(sentence[i+1])

                ### Add 1 to the transition mations at (y,x)
                T[wordIndexY, wordIndexX] +=1

                ### add last for to be connected to STOP
                if i ==len(sentence)-2:
                    T[ n-1, wordIndexY] +=1


        T[n-1,n-1] = 1
        f.close()

        ### normilzie the matrix
        for j in range(n):
            if (np.sum(T[:,j]) == 0):
                print("j: " + str(j))
                print(len(T[0]))
            T[:,j] /= np.sum(T[:,j])

        ###initilizer for Markov
        self.A = T
        m, self.n = self.A.shape
        if not np.allclose(np.sum(self.A, axis=0), np.ones(self.n)):
            raise ValueError("Not column Stochastic")
        self.labelDict = {}
        for i in range(len(words)):
            self.labelDict[words[i]] = i


     # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            yoda = SentenceGenerator("yoda.txt")
             print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        ### return the path
        path = self.path('$tart', '$top')
        path.pop()
        path.pop(0)
        return path

def sampleCode():
    A = np.array([[.7,.6],[.3,.4]])
    M = MarkovChain(A, ["hot","cold"])
    #print(M.transition("hot"))
    print(M.path("hot", "hot"))
    print(M.walk('hot', 1000).count('hot')/1000 )

    print(M.steady_state())
    yoda = SentenceGenerator("yoda.txt")
    print(yoda.babble())
if __name__ == '__main__':
    #sampleCode()
    pass

