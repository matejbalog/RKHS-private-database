import numpy as np

from DP import Laplace_mechanism, Gaussian_mechanism
from utils import print_progress

EPS = 1e-5 # for avoiding rounding errors


def compute_basis(K_zz):
    """ Compute orthonormal basis of H_M using Gram-Schmidt """

    # Compute the basis in the form of scalars A_{fm}, where
    # A_{fm} is the coefficient of k(z_m, .) in the expansion of e_f
    M = len(K_zz)
    A = np.zeros((M, M))
    squared_norms = np.zeros(M)
    K_zz_A = np.zeros((M, M))

    # Step 1: Gram-Schmidt ortogonalization
    for f in range(M):
        A[f][f] = 1
        for j in range(f):
            # project k(z_f, .) onto k(b_j, .) and get the coefficients of z's
            # proj_{b_j}(z_f) = < k(z_f, .), b_j > / norm(b_j) * b_j
            # b_j = A[j][0] k(z_0,.) + ... + A[j][j] k(z_j,.)
            denominator = squared_norms[j]
            if denominator > 0 + EPS:
                # Non-modified GS: compute indicator(f) . K_zz . A[j]
                #numerator = K_zz_A[j][f]
                # Modified GS (more stable): compute A[f] . K_zz . A[j]
                numerator = A[f].dot(K_zz_A[j])
                A[f] -= numerator / denominator * A[j]

        # Compute the pre-multiplication by the K_zz
        K_zz_A[f] = K_zz.dot(A[f])

        # Compute the squared norm of the constructed vector
        squared_norms[f] = A[f].dot(K_zz.dot(A[f]))

        # Print progress
        if ((f+1) % 100 == 0) or (f+1 == M):
            print_progress('[Progress] f = %d' % (f+1))
    print('')

    # Step 2: Gram-Schmidt normalisation
    for f in range(M):
        squared_norm = squared_norms[f]
        if squared_norm > 0 + EPS:
            A[f] /= np.sqrt(squared_norm)

    # Filter non-zero vectors, to obtain basis
    B = []
    parents = []
    for f in range(M):
        if squared_norms[f] > 0 + EPS:
            B.append(A[f])
            parents.append(f+1) # 1-based indexing for parents
    B = np.array(B)

    return B, parents


def reweight_public_compute_alpha(K_zx_rmeans, K_zz):
    # Compute basis of the RKHS subspace spanned by feature maps of public data points
    B, parents = compute_basis(K_zz)

    # Project empirical KME onto the computed basis
    alpha = B.dot(K_zx_rmeans)

    return B, alpha, parents


def reweight_public_privatize_alpha(K_xx_mean, K_zx_rmeans, K_zz, B, alpha, N_private, epsilon, delta):
    # Initialise results
    results = {'alpha': alpha}

    # Differential privacy
    if delta > 0:
        L2_sensitivity = 2.0 / N_private
        beta = alpha + Gaussian_mechanism(len(alpha), L2_sensitivity, epsilon, delta)
    if delta == 0:
        L1_sensitivity = 2.0 / np.sqrt(N_private)
        beta = alpha + Laplace_mechanism(len(alpha), L1_sensitivity, epsilon)
    results['beta'] = beta

    # Weights of the private data set
    w = beta.dot(B)
    results['weights'] = w

    # Report RKHS distance between empirical KME and its privatized projection
    dist2 = w.dot(K_zz.dot(w)) - 2.0 * w.dot(K_zx_rmeans) + K_xx_mean
    dist = np.sqrt(dist2)
    results['RKHS_dist_alg1'] = dist

    return results
