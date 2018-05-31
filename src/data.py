import argparse
import numpy as np

from datetime import datetime

from utils import json_dump


def generate_data_mixture_of_Gaussians(N_private, D):
    C = 10
    mu_means = 100.0 * np.ones(D)
    sigma_means = 200.0
    sigma_data = 30.0
    mus = np.random.normal(loc=mu_means, scale=sigma_means, size=(C, D))
    ws = np.array([1.0 / n for n in range(1, C+1)])
    ws /= np.sum(ws)
    memberships = np.random.choice(range(C), size=N_private, p=ws)
    X_private = []
    for n in range(N_private):
        mu = mus[memberships[n]]
        sample = np.random.normal(loc=mu, scale=sigma_data)
        X_private.append(sample)
    X_private = np.array(X_private)
    return X_private


def main(args_dict):
    np.random.seed(0)

    # Extract parameters
    N = args_dict['N']
    D = args_dict['D']

    # Generate the private data
    X_private = generate_data_mixture_of_Gaussians(N, D)

    # Save the generated data
    dataset = 'mixture_of_Gaussians'
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path_save = '../data/%s_N%d_D%d' % (dataset, N, D)
    np.savez(path_save + '.npz', X_private=X_private)
    json_dump({'dataset': dataset, 'N': N, 'D': D}, path_save + '.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='Number of data points to generate')
    parser.add_argument('D', type=int, help='Dimensionality of the generated data')
    args_dict = vars(parser.parse_args())
    main(args_dict)
