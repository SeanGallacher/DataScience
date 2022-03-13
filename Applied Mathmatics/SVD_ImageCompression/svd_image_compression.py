# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""
from scipy import linalg as la
from scipy import sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from imageio import imread
# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    ###Comute and return the compact SVD
    m, n = A.shape

    ## get eigen vectors
    eigs, vectors = la.eig(A.conj().T@A)
    indexs = np.flip(np.argsort(eigs))
    eigs = eigs[indexs]

    #print(eigs)
    ##Reduce the size of the 3 matricies
    vectors = vectors[:,indexs]
    sigma = np.sqrt(eigs)
    r = len(sigma[sigma > tol])
    #print(r)
    sigma = sigma[:r]
    V = vectors[:, :r]
    U = (A@V)/sigma
    #print(U)
    #print(V.conj().T)
    #print(sigma)
    return U, sigma, V





# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    ### Calcuate a basic circle
    s = 200
    Circle = np.zeros((2,s +1))
    for i in range(s  +1):
        Circle[0][i] = np.cos((np.pi * 2)*(i/s))
        Circle[1][i] = np.sin((np.pi * 2)*(i/s))
    E = np.array([[1,0,0],[0,0,1]])

    U, Sig, V = la.svd(A)
    Sig = np.diag(Sig)

    ###PLOT THE CIRCLES
    f, axarr = plt.subplots(1, 4)
    axarr[0].plot(Circle[0], Circle[1], linewidth=1)
    axarr[0].plot(E[0],E[1])
    axarr[0].axis("equal")
    axarr[0].set_title("Circle")

    Circle1 = V.conj().T@Circle
    E1 = V.conj().T@E
    axarr[1].plot(Circle1[0], Circle1[1], linewidth=1)
    axarr[1].plot(E1[0],E1[1])
    axarr[1].axis("equal")
    axarr[1].set_title("Circle : V^H ")

    Circle2 = Sig@V.conj()@Circle
    E2 = Sig@V.conj().T @ E
    axarr[2].plot(Circle2[0], Circle2[1], linewidth=1)
    axarr[2].plot(E2[0],E2[1])
    axarr[2].axis("equal")
    axarr[2].set_title("Circle : Sig V^H ")

    Circle3 = U@Sig@V.conj()@Circle
    E3 = U@Sig@V.conj().T @ E
    axarr[3].plot(Circle3[0], Circle3[1], linewidth=1)
    axarr[3].plot(E3[0],E3[1])
    axarr[3].axis("equal")
    axarr[3].set_title("Circle : U Sig V^H ")

    plt.show()

# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    ###TRIM and calculate the bst SVD aproximation
    U, Sig, V = la.svd(A)
    if s > np.count_nonzero(Sig): raise ValueError("s is too Big")
    Us = U[:,:s]
    Vs = V[:s,:]
    Sigs = Sig[:s]
    As = Us@np.diag(Sigs)@Vs

    return As, (Us.size + Vs.size + Sigs.size)



# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, Sig, V = la.svd(A, full_matrices=False)
    if err < np.min(Sig): raise ValueError("err is too small")
    ### go through Sigma from larger to smallest and select the sigma
    s = np.argmax(Sig < err)
    return svd_approx(A, s)







# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename) / 255.0

    if len(image.shape) == 3:

        ###PLOT COLOR IMAGE
        imageCom1, n1 = svd_approx(image[:,:,0],s)
        imageCom2, n2 = svd_approx(image[:, :, 1], s)
        imageCom3, n3 = svd_approx(image[:, :, 2], s)
        imageCom = np.dstack((imageCom1,imageCom2,imageCom3))
        image[image > 1] = 1
        image[image < 0] = 0
        plt.suptitle(str(image.size - (n1+n2+n3)))
        plt.subplot(121)
        plt.imshow(image)
        plt.title("image Original" )
        plt.subplot(122)
        plt.title("image Compressed")
        plt.imshow(imageCom)
        plt.show()
    else:
        ##PLOT GRAY IMAGE

        imageCom, n = svd_approx(image,s)
        plt.suptitle(str(image.size - (n)))
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title("image Original")
        plt.subplot(122)
        plt.title("image Compressed")
        plt.imshow(imageCom, cmap='gray')
        plt.show()


if __name__ == '__main__':
    #compact_svd(svd_approx(np.random.random((10,10)),5)[0])
    #A = np.random.random((3,3))

    #print(la.svd(A))
    #print("A")
    #print(compact_svd(A))
    #A = np.array([[3,1],[1,3]])
    #visualize_svd(A)

    #A = np.random.random((100, 100))
    #print(svd_approx(A, 50))
    #As, size = svd_approx(A, 23)
    A = np.array([[0.02764439, 0.20639748, 0.02910973, 0.38083646, 0.30958458,
                   0.513182, 0.88570554],
                  [0.25209254, 0.26496891, 0.06568809, 0.78616029, 0.12334944,
                   0.98270006, 0.93856434],
                  [0.19990632, 0.17554164, 0.5360761, 0.15729793, 0.96441805,
                   0.9791833, 0.37508325],
                  [0.62606431, 0.3334393, 0.11697637, 0.75925572, 0.16936398,
                   0.94690985, 0.65612168],
                  [0.32102407, 0.60436832, 0.01094071, 0.27996185, 0.05001888,
                   0.57591944, 0.2445419]])
    #print(compact_svd(A))
    #print(lowest_rank_approx(A, 1))
    #print(np.linalg.matrix_rank(As))
    #compress_image('hubble.jpg', 20)


