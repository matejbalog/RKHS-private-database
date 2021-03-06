import argparse
import numpy as np
import torch
dtype = torch.DoubleTensor
import torch.optim

from bisect import bisect
from sklearn.linear_model import Lasso
from torch.autograd import Variable

from DP import Gaussian_mechanism
from kernels import EQKernel, compute_mean_feature_vector
from RKHS import reweight_public_compute_alpha, reweight_public_privatize_alpha
from utils import json_load, json_dump, print_progress, dedup_consecutive


def experiment_alg1(K_xx_mean, K_zx_rmeans, K_zz, N_private, epsilons, delta):
    # Construct values of M (#'s of public data points) to evaluate
    Ms = [int(np.exp(e)) for e in np.linspace(0, np.log(len(K_zz)), 100)] + [len(K_zz)]
    Ms = dedup_consecutive(Ms)

    # Reweight the public data using a chosen kernel and privacy level
    B, alpha, parents = reweight_public_compute_alpha(K_zx_rmeans, K_zz)
    dists_base = []
    dists_proj = []
    dists_alg1 = [[] for _ in epsilons]
    for M in Ms:
        # Compute number of basis vectors generated by M public data points
        num_dimensions = bisect(parents, M)
        alpha_part = alpha[:num_dimensions]
        B_part = B[:num_dimensions, :M]
        K_zz_part = K_zz[:M, :M]
        K_zx_rmeans_part = K_zx_rmeans[:M]

        # Report RKHS distance between empirical KME and its projection
        w = alpha_part.dot(B_part)
        dist2_proj = w.dot(K_zz_part.dot(w)) - 2.0 * w.dot(K_zx_rmeans_part) + K_xx_mean
        dist_proj = np.sqrt(dist2_proj)
        dists_proj.append(dist_proj)

        # Report RKHS distance between emprical KME and Algorithm 1 output (for different epsilons)
        for ei, epsilon in enumerate(epsilons):
            results = reweight_public_privatize_alpha(K_xx_mean, K_zx_rmeans_part, K_zz_part, B_part, alpha_part, N_private, epsilon, delta)
            dists_alg1[ei].append(results['RKHS_dist_alg1'])

        # Compute baseline distance: private KME <-> public KME
        dist2_base = np.mean(K_zz_part) - 2.0 * np.mean(K_zx_rmeans_part) + K_xx_mean
        dist_base = np.sqrt(dist2_base)
        dists_base.append(dist_base)

        # Print progress
        if (M % 10 == 0) or (M == Ms[-1]):
            print_progress('[Progress] M: %d / %d' % (M, Ms[-1]))
    print('')

    # Return Algorithm 1 experiment results
    return {
        'Ns_public': Ms,
        'dists_proj': dists_proj,
        'dists_alg1': dists_alg1,
        'dists_base': dists_base,
    }


def experiment_rf(X_private, kernel, K_xx_mean, J, M, Z_initial, num_iters, lr, lasso_alpha, epsilons, delta):
    # Parameters
    N, D = np.shape(X_private)
    alpha = lasso_alpha / (2*N*J)   # Lasso regression regularization for w's

    # Get random features
    rf_np, rf_torch = kernel.get_random_features(J)

    # Compute random features of private data
    empirical_private = compute_mean_feature_vector(X_private, rf_np, J)

    # Number of points synthetised after which to report RKHS distances
    Ms_report = [int(10**e) for e in np.arange(0, np.log10(M+1), 0.25)]
    Ms_report = dedup_consecutive(Ms_report)

    # Iterate through epsilons
    dists_alg2 = [[] for _ in epsilons]
    for ei, epsilon in enumerate(epsilons):

        # Add privatizing noise
        L2_sensitivity = 2.0 / N
        empirical_public = empirical_private + Gaussian_mechanism(J, L2_sensitivity, epsilon, delta)

        # Run reduced set method "iterated approximate pre-images" (iteratively
        # construct M synthetic points) to approximate empirical_public
        Z = np.zeros((M, D))
        betas = []
        Psi = empirical_public
        RKHS_distances = [np.sqrt(Psi.dot(Psi))]
        for m in range(M):
            report = (m+1) in Ms_report

            # Initialize and optimise location z with Torch
            zTorch = Variable(torch.from_numpy(Z_initial[m]), requires_grad=True)
            PsiTorch = Variable(torch.from_numpy(Psi), requires_grad=False)
            optimizer = torch.optim.Adam([zTorch], lr=lr)
            for i in range(num_iters):
                optimizer.zero_grad()
                phiz = rf_torch(zTorch.view(1, D)).view(-1) # -1 equals J or 2*J
                loss = - PsiTorch.dot(phiz) ** 2 / phiz.dot(phiz)
                # Backward pass and gradient step
                loss.backward()
                optimizer.step()
            z = zTorch.data.numpy()

            # Compute optimal weight beta and update residual vector
            phiz = rf_np(z)
            beta = Psi.dot(phiz) / phiz.dot(phiz)
            Psi = Psi - beta * phiz

            # Save found synthetic point and its optimal weight
            Z[m] = z
            betas.append(beta)
            RKHS_distances.append(np.sqrt(Psi.dot(Psi)))

            if report:
                # Compute optimal reweighting
                Z_np = Z[:(m+1)]
                Phi = rf_np(Z_np)
                clf = Lasso(fit_intercept=False, alpha=alpha)
                clf.fit(np.transpose(Phi), empirical_public)
                w_np = np.reshape(clf.coef_, (m+1)) # reshape needed for case m = 0
                
                # Compute real distance from private embedding
                K_zz = kernel.get_kernel_matrix(Z_np)
                K_zx_rmeans = kernel.get_kernel_matrix_rowmeans(Z_np, X_private)
                dist2 = w_np.dot(K_zz.dot(w_np)) - 2.0 * w_np.dot(K_zx_rmeans) + K_xx_mean
                dist_k_opt = np.sqrt(dist2)
                print('[Progress] epsilon=%f, M=%d, ||.-.|| = %f' % (epsilon, m+1, dist_k_opt))
                dists_alg2[ei].append(dist_k_opt)

    # Return experiment results
    return {
        'J': J,
        'M': M,
        'num_iters': num_iters,
        'lr': lr,
        'Ms_report': Ms_report,
        'dists_alg2': dists_alg2,
    }


def main(args_dict):
    # Fix randomness
    np.random.seed(0)
    torch.manual_seed(0)

    # Extract parameters from the command line
    dataset = args_dict['dataset']
    algorithm = int(args_dict['algorithm'])
    assert algorithm in [1, 2]
    public_data = args_dict['public_data']
    assert public_data in ['leak', 'random']

    # Load the private data
    data = np.load(dataset + '.npz')
    X_private = data['X_private']
    N_private, D = np.shape(X_private)
    print('[OK] Loaded N=%d private data points with dimension D=%d' % (N_private, D))

    # Construct the public data (M data points)
    M = args_dict['M']
    if public_data == 'leak':
        # Publishable subset => take the first M private data points
        X_public = X_private[:M]
    if public_data == 'random':
        # No publishable subset => sample synthetic data points at random
        mu_public = np.zeros(D)
        sigma_public = args_dict['sigma_public']
        X_public = np.random.normal(loc=mu_public, scale=sigma_public, size=(M, D))

    # Get the kernel
    lengthscale = 100.0 * np.sqrt(D)
    gamma = 1.0 / lengthscale ** 2
    kernel = EQKernel(D, gamma)

    # Load or compute kernel matrix mean K_xx_mean
    j = json_load(dataset + '.json')
    key = 'kernel_mean_' + str(kernel)
    if key in j:
        K_xx_mean = j[key]
        print('[OK] Found mean of K_xx = %f' % (K_xx_mean))
    else:
        K_xx_mean = kernel.get_kernel_matrix_mean(X_private)
        j[key] = K_xx_mean
        json_dump(j, dataset + '.json')
        print('[OK] Computed and saved mean of K_xx = %f' % (K_xx_mean))

    # Privacy requirement
    epsilons = args_dict['epsilons']
    delta = args_dict['delta']

    # Run the experiment, obtaining results as a dictionary
    if algorithm == 1:
        K_zx_rmeans = kernel.get_kernel_matrix_rowmeans(X_public, X_private)
        K_zz = kernel.get_kernel_matrix(X_public)
        results = experiment_alg1(K_xx_mean, K_zx_rmeans, K_zz, N_private, epsilons, delta)
    if algorithm == 2:
        J = args_dict['J']
        Z_initial = X_public
        lasso_alpha = args_dict['lasso_alpha']
        num_iters = args_dict['num_iters']
        lr = args_dict['lr']
        results = experiment_rf(X_private, kernel, K_xx_mean, J, M, Z_initial, num_iters, lr, lasso_alpha, epsilons, delta)

    # Save experiment results
    results.update({
        'dataset': dataset,
        'algorithm': algorithm,
        'public_data': public_data,
        'N_private': N_private,
        'D': D,
        'kernel': str(kernel),
        'epsilons': epsilons,
        'delta': delta,
        })
    path_save = '../results/D%d_alg%d_%s_M%d.json' % (D, algorithm, public_data, M)
    json_dump(results, path_save)
    print('[OK] Saved Algorithm %d experiment results with "%s" public data to:\n\t%s' % (algorithm, public_data, path_save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Options common to both Algorithm 1 and Algorithm 2
    parser.add_argument('dataset', type=str, help='dataset name')
    parser.add_argument('public_data', type=str, help='source of (initial) public data ("leak" or "random")')
    parser.add_argument('--M', default=100, type=int, help='Number of synthetic data points to use')
    parser.add_argument('--sigma_public', default=500.0, type=float, help='Standard deviation of spherical Gaussian synthetic data point sampling distribution')
    parser.add_argument('--epsilons', default=[0.01, 0.1, 1.0], type=float, nargs='+', help='Epsilon privacy parameters to use')
    parser.add_argument('--delta', default=0.000001, type=float, help='Delta privacy parameter')

    subparsers = parser.add_subparsers(dest='algorithm')
    sp = subparsers.add_parser('1')
    sp = subparsers.add_parser('2')
    sp.add_argument('--J', default=10000, type=int, help='Number of random features')
    sp.add_argument('--num_iters', default=100, type=int, help='Iterations for each synthetic data point location')
    sp.add_argument('--lr', default=100.0, type=float, help='Learning rate used within Algorithm 2')
    sp.add_argument('--lasso_alpha', default=100.0, type=float, help='Lasso alpha regularization strength for weight optimisation in Algorithm 2')

    args_dict = vars(parser.parse_args())
    main(args_dict)
