# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name
<Class
<Date
"""

import numpy as np
import scipy
import scipy.linalg
from matplotlib import pyplot as plt

from scipy import linalg as la
from scipy import spatial
from scipy import stats
from scipy.linalg import lu_factor
from scipy import sparse
from scipy.sparse import linalg as spla
from datetime import datetime

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    minD = 10**200
    mixX = None
    ### find the minimum point by looking at all nodes
    for x in X:
        d = la.norm((x-z))
        if d < minD:
            minD = d
            minX = x
    return minX, minD

# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if type(x) != np.ndarray:
            raise TypeError("Not an np array")
        #initialize variables
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None
        self.size = 0

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        def find_and_insert(data, curNode, curPiv):
            if (data == curNode.value).all():
                raise ValueError("Cant add duplicates")

            ## go right
            if curNode.value[curPiv] <= data[curPiv]:
                curPiv += 1
                curPiv = curPiv % self.k
                ### if you hit the end of the tree, insert
                if curNode.right == None:
                    curNode.right = KDTNode(data)
                    curNode.right.pivot = curPiv
                    self.size += 1
                    return
                ## keep looking
                find_and_insert(data, curNode.right, curPiv)
            ## go left
            else:
                curPiv += 1
                curPiv = curPiv % self.k
                ### if you hit the end of the tree, insert
                if curNode.left == None:
                    curNode.left = KDTNode(data)
                    curNode.left.pivot = curPiv
                    self.size+=1
                    return
                ## keep looking
                find_and_insert(data, curNode.left, curPiv)
        newNode = KDTNode(data)
        ### If tree is empty
        if self.size == 0:
            self.root = newNode
            self.k = len(data)
            newNode.pivot = 0
            self.size = 1

        ##else instert into the tree
        else:
            find_and_insert(data,self.root, 0)

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """

        def d(x,z):
            return la.norm(x,z)
        def KDSearch(curNode, nearNode, dStar):
            if curNode == None:
                return nearNode, dStar
            x = curNode.value
            i = curNode.pivot
            ##compare the curent node to current min node
            if d(x,z) < dStar:
                nearNode = curNode
                dStar = d(x,z)
            ###look left
            if z[i] < x[i]:
                nearNode, dStar = KDSearch(curNode.left, nearNode, dStar)
                if z[i] + dStar == x[i]:
                    nearNode, dStar = KDSearch(curNode.right, nearNode, dStar)
            ##look right
            else:
                nearNode, dStar = KDSearch(curNode.right, nearNode, dStar)
                if z[i] - dStar <= x[i]:
                    nearNode, dStar = KDSearch(curNode.left, nearNode, dStar)
            ### return the nearest neighbor
            return nearNode, dStar
        node, dStar = KDSearch(self.root, self.root, d(self.root.value, z))
        return dStar, node.value


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        ### set up varaibles
        self.n_neighbors = n_neighbors
        self.tree = scipy.spatial.KDTree
        #scipy.stats.mode()

    def fit(self, M, labels):
        ###Load the tree and fit the data
        m, n = M.shape
        self.tree = scipy.spatial.KDTree(M)
        self.labels = labels

    def predict(self, z):
        ###find the nearest neighbor and predict its label
        dis, ind = self.tree.query(z, self.n_neighbors)
        most_common = scipy.stats.mode(self.labels[ind])
        return most_common[0]



# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    ##load the data
    data = np.load(filename)

    X_train = data["X_train"].astype(np.float)  # Training data
    print(X_train)
    print(type(X_train))
    print(X_train.shape)
    y_train = data["y_train"]  # Training labels
    print(y_train)
    print(type(y_train))
    print(y_train.shape)
    X_test = data["X_test"].astype(np.float)  # Test data
    y_test = data["y_test"]  # Test labels
    ### Train the data
    classifyX = KNeighborsClassifier(n_neighbors)
    classifyX.fit(X_train, y_train)

    ### predict the data
    numCor = 0
    for i in range(len(y_test)):
        value = classifyX.predict(X_test[i])
        if y_test[i] == value:
            numCor +=1

    ##show the image
    #plt.imshow(X_test[0].reshape((28,28)), cmap="gray")
    #plt.show()

    ## return the accuracy
    return numCor/len(y_test)


####run test cases
def testProb1():
    X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    z = np.array([1,2,4])
    print(exhaustive_search(X,z))
    assert(exhaustive_search(X,z) == [1,2,3], la.norm(X[0]-z))


def testKDT():
    list1 = [np.array([3,1,4]),np.array([1,2,7]),np.array([4,3,5]),np.array([2,0,3]),
             np.array([2,4,5]),np.array([6,1,4]),np.array([1,4,3]),
             np.array([0,5,7]),np.array([5,2,5])]
    kdt = KDT()
    for a in list1:
        kdt.insert(a)

    list2 = np.array([[5, 5], [2, 2],
    [8, 8],
     [3, 3],
     [4, 4],
     [1, 1],
     [6, 6],
     [7, 7],
     [9, 9]])
    kdt2 = KDT()
    for i in range(len(list2)):
        kdt2.insert(list2[i])
    print(kdt2)
    list3 = np.array([[2 ,3],[1,4]])
    kdt3 = KDT()
    for i in range(len(list3)):
        kdt3.insert(list3[i])

    list4 = np.array([[2 ,3 ,4],
     [5, 6 ,7],
     [7 ,1 ,9],
     [3, 4 ,8]])
    kdt4 = KDT()
    for i in range(len(list4)):
        kdt4.insert(list4[i])
    assert (kdt4.__str__() == 'KDT(k=3)\n[2 3 4]\tpivot = 0\n[5 6 7]\tpivot = 1\n[7 1 9]\tpivot = 2\n[3 4 8]\tpivot = 0')
    assert(kdt3.__str__() == 'KDT(k=2)\n[2 3]\tpivot = 0\n[1 4]\tpivot = 1' )
    assert(kdt2.__str__() == 'KDT(k=2)\n[5 5]\tpivot = 0\n[2 2]\tpivot = 1\n[8 8]\tpivot = 1\n[1 1]\tpivot = 0\n[3 3]\tpivot = 0\n[6 6]\tpivot = 0\n[9 9]\tpivot = 0\n[4 4]\tpivot = 1\n[7 7]\tpivot = 1')
    #kdt4.query(np.array([[5,3,32]]))


def testFitPredict():
    data = np.random.random((100, 5))  # 100 5-dimensional points.
    target = np.random.random(5)
    tree = scipy.spatial.KDTree(data)
    # Query the tree for the 3 nearest neighbors.
    distances, indices = tree.query(target, k=3)
    classifier = KNeighborsClassifier(3, )
    classifier.fit(data, target)
    #print(classifier.predict(target))
    #print(indices)


def testProb6():
    print(prob6(4))
    #assert(prob6(4, "mnist_subset.npz")>.9)

if __name__ == '__main__':
    #testProb1()
    #testKDT()
    #testFitPredict()
    pass
