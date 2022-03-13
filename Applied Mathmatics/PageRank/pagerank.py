# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        m, n = A.shape


        ### format colums
        sums = np.zeros(n)
        for i in range(n):
            sumC = np.sum(A[:,i])
            if sumC == 0:
                sumC = n
                A[:, i] = np.ones(n)
            sums[i] = sumC
        self.Ahat = A/sums.T
        if labels == None:
            labels = [i for i in range(n)]
        if len(labels)!= m:
            raise ValueError('label length and graph length are different')
        self.labels = labels
        self.n = n


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        p = la.solve(np.eye(self.n) - epsilon*self.Ahat, np.ones(self.n)*(1-epsilon)/self.n)

        return dict(zip(self.labels, p))

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """

        B =(epsilon*self.Ahat + np.ones((self.n, self.n))*(1-epsilon)/self.n)
        eigs, eigvects = la.eig(B)
        i = np.argmax(np.real(eigs))
        p = eigvects[:,i]/np.sum(eigvects[:,i])
        return dict(zip(self.labels, p))


    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        p0= np.array([1/self.n for i in range(self.n)])
        k = 1
        pk = self.Ahat@p0
        while k < maxiter and la.norm(p0-pk) > tol:
            p0 = pk
            B = (epsilon * self.Ahat + np.ones((self.n, self.n)) * (1 - epsilon) / self.n)
            pk = B@p0
            k+=1
        return dict(zip(self.labels, pk))





# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    return [k for k,v in sorted(d.items(), reverse=True, key=lambda item: item[1])]

def o():
    A = np.array([[0,0,0,0],[1,0,1,0],[1,0,0,1],[1,0,1,0]])
    DG = DiGraph(A, ['a','b','c','d'])
    print(DG.linsolve())
    print(DG.itersolve(maxiter=500))
    print(DG.eigensolve())
    print(get_ranks(DG.linsolve()))


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    data = {}
    enumDict = {}
    allValues = set()
    for i, line in enumerate(Lines):
        info = line.strip().split('/')
        info[-1] = info[-1].strip()
        data[info[0]] = info[1:]
        for v in info:
            allValues.add(v.strip())
        enumDict[info[0]] = i
    n = len(allValues)
    M = np.zeros((n,n))
    otherValues = {}
    ### v is the location in the matrix
    for k, v in enumDict.items():
        for o in data[k]:
            ###arrows point at row values,
            if o not in enumDict.keys():
                otherValues[o] = len(enumDict) + len(otherValues)
                M[int(otherValues[o]),int(v)] +=1
            else:
                M[int(enumDict[o]),int(v)] +=1

    labels = list(enumDict.keys()) + list(otherValues.keys())
    DG = DiGraph(M, labels=labels)

    rankingD = DG.itersolve(epsilon=epsilon)
    rankingD = {k:v for k,v in sorted(rankingD.items(), reverse=True, key=lambda item: item[0])}
    ranks = get_ranks(rankingD)

    return ranks





# Problem 5
def rank_ncaa_teams(filename='ncaa2010.csv', epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    data = {}
    allTeams = set()

    with open(filename, 'r') as infile:
        games = [l.split(',') for l in infile.read().strip().split('\n')][1:]
    ### put data  of winners to losers in a dictionary


    labelDict = {}
    for pair in games:
        for team in pair:
            allTeams.add(team)
    allTeams = sorted(allTeams)



    ### create label dict
    for i, v in enumerate(allTeams):
        labelDict[v] = i

    n = len(allTeams)
    M = np.zeros((n,n))

    ### v is the location in the matrix
    for k, v in games:
        ###arrows point at row values,
        winner = int(labelDict[k])
        loser = int(labelDict[v])
        M[winner,loser] += 1

    labels = list(labelDict.keys())
    DG = DiGraph(M, labels=labels)

    rankingD = DG.itersolve(epsilon=epsilon)
    ranks = get_ranks(rankingD)

    return ranks



# Problem 6
import networkx as nx
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    file1 = open(filename, 'r', encoding="utf-8" )
    Lines = file1.readlines()

    data = {}
    labelDict = {}
    allActors = set()
    if False:
        with open(filename, 'r') as infile:
            MovieActors = [l.split('/') for l in infile.read().split('\n')]

        data = {k[0]:k[1:] for k in MovieActors}

        labelDict = {}
        for actors in data.values():
            for actor in actors:
                allActors.add(actor)
    if True:
        for i, line in enumerate(Lines):
            info = line.strip().split('/')
            info[-1] = info[-1].strip()
            data[info[0].strip()] = info[1:]
            ### add all actors to set
            for v in info[1:]:
                allActors.add(v.strip())
        ###create labelDict
        for i, v in enumerate(allActors):
            labelDict[v] = i
    n = len(allActors)

    M = np.zeros((n, n))

    ### v is the location in the matrix

    labels = list(labelDict.keys())
    DG = nx.DiGraph()

    for label in allActors:
        DG.add_node(label)
    for k, v in data.items():
        for i, a in enumerate(data[k][:-1]):
            for b in data[k][i + 1:]:
                if not DG.has_edge(b,a):
                    DG.add_edge(b,a, weight=0)
                DG[b][a]['weight'] += 1

    rankingD = nx.pagerank(DG, alpha=epsilon)

    #rankingD = {k: v for k, v in sorted(rankingD.items(), reverse=True, key=lambda item: item[0])}
    ranks = get_ranks(rankingD)

    return ranks

import time
if __name__ == '__main__':
    pass
    #t = time.time()
    #print(rank_actors())
    if False:
        print(time.time()-t)
        print(rank_websites(epsilon=0.14))
        print(rank_websites(epsilon=0.06))
        print(rank_websites(epsilon=0.92))
        print(rank_websites(epsilon=0.27))
        print(rank_websites(epsilon=0.1))
    if False:
        A =np.array(
        [[1, 1, 0, 1],
         [1 ,1, 0 ,1],
        [0,
        1,
        1,
        1],
        [1 ,1, 1 ,1]])
        DG = DiGraph(A)
        print(DG.eigensolve())
    print(rank_actors(epsilon=.16))
