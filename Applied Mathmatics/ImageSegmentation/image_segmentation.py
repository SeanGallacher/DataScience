# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np

from scipy import linalg as la
from scipy import sparse as sp
import scipy.sparse.linalg as spla
from imageio import imread
from matplotlib import pyplot as plt

def Trylap():
    pass

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    diag = []
    for i in range(len(A)):
        diag.append(sum(A[:, i]))
    D = np.diag(np.array(diag))
    return D-A



# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    eigs = np.real(la.eigvals(L))
    eigs = np.sort(eigs)
    zeros = [v for v in eigs if abs(v) < tol]
    return len(zeros), eigs[1]



# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename)/255.0
        self.brightness = self.image
        self.Color = False
        if len(self.image.shape) == 3:
            self.brightness = self.image.mean(axis=2)
            self.Color = True
        self.flatImage = self.image.flatten()
        self.flatBright = self.brightness.flatten()
        self.p = 0

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if len(self.image.shape) == 3:
            plt.imshow(self.image)
            plt.show()
        else:
            plt.imshow(self.image, cmap='gray')
            plt.show()


    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        if self.Color:
            m, n, p = self.image.shape
            self.p = p
        else:
            m, n = self.image.shape
            self.p = 0
        self.m = m
        self.n = n
        size = m*n
        A = sp.lil_matrix((size,size))
        self.D = []
        ### get weight between i and j
        def getWeight(i,j, dist):
            term1 = (self.flatBright[i] -self.flatBright[j])/(sigma_B2)
            term1 = np.abs(term1)
            term2 = dist/(sigma_X2)
            return np.exp(-term1-term2)

        D = np.zeros(m * n)
        ### assign weights to A
        for i in range(len(self.flatBright)):
            neighbors, distances = get_neighbors(i,r,m,n)
            sumWeight = 0
            for k in range(len(neighbors)):
                j = neighbors[k]
                dist = distances[k]
                weight = getWeight(i, j, dist)
                A[i, j] = weight
                sumWeight += weight
            D[i] = sumWeight

        A = A.tocsc()

        #print(D)

        return A, D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask thaat segments the image."""
        L = sp.csgraph.laplacian(A)
        D1d2 = sp.diags([1/np.sqrt(d) for d in D])

        B = (D1d2@(L@D1d2))
        #print(B)
        eigsVector = spla.eigsh(B, which='SM', k=2)[1][:,1]
        #print(eigsVector)
        maskVector = np.reshape(eigsVector, (self.m, self.n))
        mask = maskVector > 0

        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r,sigma_B,sigma_X)
        mask = self.cut(A,D)
        if self.Color:
            if self.p == 4:
                mask = np.dstack((mask, mask, mask,mask))
            if self.p == 3:
                mask = np.dstack((mask, mask, mask))
        maskN = ~mask
        Postiveimg = self.image * mask
        Negativeimg = self.image * maskN

        if len(self.image.shape) == 3:
            plt.figure()
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(self.image)
            axarr[1].imshow(Postiveimg)
            axarr[2].imshow(Negativeimg)
            plt.show()
        else:
            plt.figure()
            f, axarr = plt.subplots(1, 3)
            axarr[0].imshow(self.image, cmap='gray')
            axarr[1].imshow(Postiveimg, cmap='gray')
            axarr[2].imshow(Negativeimg, cmap='gray')
            plt.show()


def codeToRun():
    HMA = np.load("HeartMatrixA.npz")
    HMD = np.load("HeartMatrixD.npy")

    #IS = ImageSegmenter('dream.png')
    #A,D = IS.adjacency()
    #assert(np.allclose(HMD, D))
    #print(A, D)
    #IS.segment()

    #npy_D = np.load("HeartMatrixD.npy")
    #npz_A = np.load("HeartMatrixA.npz")

    #img_segment = ImageSegmenter("blue_heart.png")
    # # img_segment.show_original()
    #A, D = img_segment.adjacency()
    #print(np.allclose(npy_D, D))
    #print(str(A) == str(A))


    A = np.array([[0,1,0,0,1,1],
                  [1,0,1,0,1,0],
                  [0,1,0,1,0,0],
                  [0,0,1,0,1,1],
                 [1,1,0,1,0,0],
                 [1,0,0,1,0,0]])
    print(connectivity(A))

    A = np.array([[0, 3, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 2, .5], [0, 0, 0, 2, 0, 1],
                  [0, 0, 0, .5, 1, 0]])
    print(connectivity(A))
    A1 = np.array([[0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 0],
                   [1, 0, 0, 1, 0, 0]])
    print(connectivity(A1))

    #####FIX GREY SCALE
    #assert (np.array_equal(HMD,D))
    #ImageSegmenter("dream.png").show_original()
    #ImageSegmenter("dream_gray.png").show_original()
    #ImageSegmenter("dream_gray.png").segment()
    #ImageSegmenter("dream.png").segment()
    #ImageSegmenter("monument_gray.png").segment()
    #ImageSegmenter("monument.png").segment()
if __name__ == '__main__':
    #codeToRun()
    pass

